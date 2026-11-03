import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    rank = n * p  # 16
    U = [[0] * rank for _ in range(n * m)]  # 8 x 16
    V = [[0] * rank for _ in range(m * p)]  # 32 x 16
    W = [[0] * rank for _ in range(p * n)]  # 16 x 16
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j][r] += 1
                V[j * p + k][r] += 1
            W[k * n + i][r] += 1
    
    return {"U": U, "V": V, "W": W}
