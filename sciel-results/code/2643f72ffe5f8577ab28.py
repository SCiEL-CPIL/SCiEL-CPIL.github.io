"""Matrix multiplication 3x3x5 rank-minimal decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

_DEADLINE = 140.0  # safety margin within 150s wall-clock
_START = time.time()

def construct():
    n, m, p = 3, 3, 5
    rank = n * m * p  # schoolbook rank 45
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Return as lists to match interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
