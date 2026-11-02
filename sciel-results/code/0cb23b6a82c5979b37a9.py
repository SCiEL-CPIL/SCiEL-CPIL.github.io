import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index
                idx_u = i * m + j
                # B index
                idx_v = j * p + k
                # C index (W[k*n+i, r])
                idx_w = k * n + i
                
                U[idx_u, r] = 1.0
                V[idx_v, r] = 1.0
                W[idx_w, r] = 1.0
                r += 1
                
    # Convert to list of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
