import numpy as np
import time

def construct():
    """
    Returns a rank-64 decomposition for 2x4 @ 4x8 matrix multiplication.
    U: (8, 64), V: (32, 64), W: (16, 64)
    """
    n, m, p = 2, 4, 8
    rank = n * m * p
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
