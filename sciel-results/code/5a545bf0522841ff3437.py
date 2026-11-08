import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    # Standard decomposition rank: n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices:
    # A: (i, k) -> i * m + k
    # B: (k, j) -> k * p + j
    # C: (i, j) -> j * n + i  (based on W[k*n+i, r] convention where k is col, i is row)
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index
                u_idx = i * m + k
                # V index
                v_idx = k * p + j
                # W index
                w_idx = j * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
