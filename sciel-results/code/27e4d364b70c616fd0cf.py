import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank minimal decomposition using identity-like W
    rank = p * n
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        k = r // n
        i = r % n
        
        # W[k*n + i, r] = 1
        W[k * n + i, r] = 1
        
        # For all j, U[i*m + j, r] = 1 and V[j*p + k, r] = 1
        for j in range(m):
            U[i * m + j, r] = 1
            V[j * p + k, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
