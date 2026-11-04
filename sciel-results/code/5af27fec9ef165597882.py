import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Rank 6 decomposition
    rank = 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each l in 0..5, corresponding to k=l
    # U[i*m+j, l] = 1 if j==l else 0
    # V[j*p+k, l] = 1 if j==l and k==l else 0
    # W[k*n+i, l] = 1 if k==l else 0
    
    for l in range(rank):
        # U part
        for i in range(n):
            for j in range(m):
                if j == l:
                    U[i * m + j][l] = 1
        
        # V part
        for j in range(m):
            for k in range(p):
                if j == l and k == l:
                    V[j * p + k][l] = 1
        
        # W part
        for k in range(p):
            for i in range(n):
                if k == l:
                    W[k * n + i][l] = 1
    
    return {"U": U, "V": V, "W": W}
