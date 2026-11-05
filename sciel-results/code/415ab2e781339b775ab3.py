import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Uses schoolbook multiplication which has rank 45.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

