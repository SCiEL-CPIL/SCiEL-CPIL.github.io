"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses block decomposition: splits the 4-dimensional inner product into two 2-dimensional subspaces.
    Each subspace corresponds to a 2x2 * 2x5 multiplication.
    The rank of 2x2 * 2x5 is 20 (5 columns * rank 4 for 2x2 * 2x1).
    Total rank = 20 + 20 = 40.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # The total rank is n*m*p = 40.
    # We construct the decomposition by treating the m=4 dimension as two blocks of size 2.
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5)
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5)
    
    rank_block = 20 # Rank for 2x2 * 2x5
    total_rank = 2 * rank_block # 40
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # Block 1: j in {0, 1}
    for i in range(n):
        for j in range(2):
            for k in range(p):
                # U index: i*m + j
                U[i*m + j, col] = 1
                # V index: j*p + k
                V[j*p + k, col] = 1
                # W index: k*n + i
                W[k*n + i, col] = 1
                col += 1
                
    # Block 2: j in {2, 3}
    for i in range(n):
        for j in range(2, 4):
            for k in range(p):
                # U index: i*m + j
                U[i*m + j, col] = 1
                # V index: j*p + k
                V[j*p + k, col] = 1
                # W index: k*n + i
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
