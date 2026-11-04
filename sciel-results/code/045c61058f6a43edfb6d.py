import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Construct a rank np decomposition
    # c = k * n + i
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            c = k * n + i
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k][c] = 1
                
    return {"U": U, "V": V, "W": W}
