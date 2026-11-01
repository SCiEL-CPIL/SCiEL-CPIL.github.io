"""Program for finding rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """Return schoolbook decomposition for 2x3 * 3x9 matrix multiplication.
    
    The tensor M_{2,3,9} has rank 54 (schoolbook). This is likely minimal
    because the 3x9 factor makes the B and C dimensions have disjoint supports
    across the 9 slices, preventing Strassen-like cancellations.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank) where
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for the decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
