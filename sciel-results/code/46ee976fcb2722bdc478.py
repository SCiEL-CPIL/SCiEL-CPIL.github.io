import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index: i*m + k
                U[i * m + k][col] = 1
                # V index: k*p + j
                V[k * p + j][col] = 1
                # W index: j*n + i (based on trace convention W[j*n+i] for C[i,j])
                W[j * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
