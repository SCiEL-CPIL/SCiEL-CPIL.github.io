import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0

def construct():
    n, m, p = DIMENSIONS
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    return {"U": U_naive, "V": V_naive, "W": W_naive}
