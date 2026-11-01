"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct the schoolbook tensor decomposition for matrix multiplication.
    For dimensions n, m, p, the rank is n*m*p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # No known rank reduction for 3x4x4 that is easily implementable and guaranteed to be lower than 48.
    # The schoolbook rank is 48.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
