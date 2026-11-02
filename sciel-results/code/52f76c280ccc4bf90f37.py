import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 63
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
