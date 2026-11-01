import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    rank = m  # = 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r] = delta(j, r)
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    # V[j*p + k, r] = delta(j, r)
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W[k*n + i, r] = 1 for all r
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
