import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    
    # The tensor T is the direct sum of p tensors T_k
    # Each T_k corresponds to multiplying A (n x m) by the k-th column of B (m x 1)
    # to get the k-th column of C (n x 1).
    # The rank of each T_k is n*m.
    # Total rank is p * n * m.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (as per prompt convention)
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
