import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        # r corresponds to j index
        j = r
        # U[i * m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j, r] = 1.0
            
        # V[j * p + k, r] = 1 for all k
        for k in range(p):
            V[j * p + k, r] = 1.0
            
        # W[k * n + i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
