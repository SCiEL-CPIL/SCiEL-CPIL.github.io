import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for l in range(rank):
        i = l // m
        j = l % m
        
        # U: A[i, j]
        # u = i * m + j
        U[i * m + j][l] = 1
        
        # V: B[j, k] for all k
        # v = j * p + k
        for k in range(p):
            V[j * p + k][l] = 1
            
        # W: C[i, k] for all k
        # w = k * n + i
        for k in range(p):
            W[k * n + i][l] = 1
            
    return {"U": U, "V": V, "W": W}
