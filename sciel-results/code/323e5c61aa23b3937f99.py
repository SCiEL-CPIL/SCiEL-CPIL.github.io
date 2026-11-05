import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 3 decomposition
    # Structure:
    # For each j in range(m), we have a component c=j.
    # U[i*m + j, c] = 1 if c == j else 0
    # V[j'*p + k, c] = 1 if c == j' else 0
    # W[k*n + i, c] = 1 for all i, k
    
    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(m):
        # U: index i*m + j. Non-zero when j == c.
        for i in range(n):
            idx_u = i * m + c
            U[idx_u, c] = 1.0
            
        # V: index j'*p + k. Non-zero when j' == c.
        for k in range(p):
            idx_v = c * p + k
            V[idx_v, c] = 1.0
            
        # W: index k*n + i. Always 1 for this component.
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                W[idx_w, c] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
