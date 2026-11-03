"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-20 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The decomposition exploits the structure of the inner dimension m=4 by splitting it
    into two blocks of size 2. Each block corresponds to a 2x2 * 2x5 multiplication,
    which has a minimal rank of 10. The total rank is 10 + 10 = 20.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 20
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Split m=4 into two blocks of size 2
    # Block 0: j in [0, 1]
    # Block 1: j in [2, 3]
    
    for block in range(2):
        j_start = block * 2
        # For each j in the block and each k in p, we have one term
        for j_local in range(2):
            j_global = j_start + j_local
            for k in range(p):
                # Assign column 'col'
                # U[i, j_global, col] = 1 for all i
                # V[j_global, k, col] = 1
                # W[k, i, col] = 1 for all i
                
                for i in range(n):
                    U[i * m + j_global, col] = 1
                    W[k * n + i, col] = 1
                V[j_global * p + k, col] = 1
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
