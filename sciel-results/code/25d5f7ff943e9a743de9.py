import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank is m (which is 3) based on the decomposition:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # U[i,j,c] = 1 if j == c else 0
    # V[j,k,c] = 1 if j == c else 0
    # W[k,i,c] = 1
    
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n):
            # U index: i*m + j. Here j=c
            U[i * m + c][c] = 1
            
        for k in range(p):
            # V index: j*p + k. Here j=c
            V[c * p + k][c] = 1
            
        for k in range(p):
            for i in range(n):
                # W index: k*n + i
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
