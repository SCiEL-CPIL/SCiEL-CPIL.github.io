import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    # Schoolbook rank
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0] * rank for _ in range(m * p)]  # 16 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
