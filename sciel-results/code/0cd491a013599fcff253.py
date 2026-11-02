import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    # c = 5*a + b, where a in {0,1,2}, b in {0,1,2,3,4}
    # U[i*m+j, c] = 1 if i == a else 0
    # V[j*p+k, c] = 1 if j == b else 0
    # W[k*n+i, c] = 1
    
    rank = 15
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        a = c // 5
        b = c % 5
        
        # Fill U
        for i in range(n):
            for j in range(m):
                if i == a:
                    U[i * m + j, c] = 1.0
        
        # Fill V
        for j in range(m):
            for k in range(p):
                if j == b:
                    V[j * p + k, c] = 1.0
        
        # Fill W
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
