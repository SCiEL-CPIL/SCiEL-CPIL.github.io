import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 2 decomposition
    rank = 2
    
    # U shape: (n*m, rank) = (6, 2)
    # V shape: (m*p, rank) = (27, 2)
    # W shape: (p*n, rank) = (18, 2)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill U
    # i=0 (rows 0 to m-1)
    for j in range(m):
        U[j, 0] = 1.0
        U[j, 1] = 0.0
    # i=1 (rows m to 2m-1)
    for j in range(m):
        U[m + j, 0] = 0.0
        U[m + j, 1] = 1.0
        
    # Fill V
    # All entries are 1
    V.fill(1.0)
    
    # Fill W
    # W index: k*n + i
    for k in range(p):
        W[k * n + 0, 0] = 1.0
        W[k * n + 0, 1] = 0.0
        W[k * n + 1, 0] = 0.0
        W[k * n + 1, 1] = 1.0
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
