import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Standard Schoolbook Decomposition
    # Rank = n * m * p
    # We use entries 1.0 to ensure the sum is exactly 1.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # where r corresponds to the triple (i, j, k).
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map (i, j) to U row
                u_idx = i * m + j
                # Map (j, k) to V row
                v_idx = j * p + k
                # Map (k, i) to W row
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
