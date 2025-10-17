#Vinícius Henrique Colere Soares - RA: 2751585

import cv2
import numpy as np
import pandas as pd

video_path = "pendulo.mp4"
output_csv = "dados_pendulo.csv"

#Abrir video e pegar o numero de fps
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Total de frames: {frame_count}")

#Tracking por cor (vermelho)
lower = np.array([0, 100, 100]) 
upper = np.array([10, 255, 255])


frames, tempos, xs, ys = [], [], [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    time = frame_number / fps  # tempo em segundos

    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    #Ajusta os contornos da maçã
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pega o maior contorno
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 3:  # ignora ruído
            frames.append(frame_number)
            tempos.append(time)
            xs.append(x)
            ys.append(y)



cap.release()
cv2.destroyAllWindows()

#Salva na planilha
df = pd.DataFrame({"frame": frames, "tempo (s)": tempos, "x (px)": xs, "y (px)": ys})
df.to_csv(output_csv, index=False)

print("salvo em", output_csv)
