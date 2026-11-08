import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Construct the standard schoolbook decomposition
    # This is a valid exact decomposition with rank = n * m * p
    rank = N * M * P
    
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
