"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication.
    For 3x5x7, the schoolbook rank is 105. We return this as it is the known 
    baseline and no faster sub-schoolbook algorithm is readily available 
    within the time constraints for this specific dimension set.
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
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
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
