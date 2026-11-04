import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        i = c // p
        k = c % p
        
        # U[i*m + j, c] = 1 for all j
        for j in range(m):
            U[i * m + j][c] = 1
            
        # V[j*p + k, c] = 1 for all j
        for j in range(m):
            V[j * p + k][c] = 1
            
        # W[k*n + i, c] = 1
        W[k * n + i][c] = 1
        
    return {"U": U, "V": V, "W": W}
