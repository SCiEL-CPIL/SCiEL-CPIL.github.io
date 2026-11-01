import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 4
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Decomposition for j=0 (ranks 0, 1)
    # r=0: i=0, j=0
    U[0 * m + 0, 0] = 1
    for k in range(p):
        V[0 * p + k, 0] = 1
        W[k * n + 0, 0] = 1
        
    # r=1: i=1, j=0
    U[1 * m + 0, 1] = 1
    for k in range(p):
        V[0 * p + k, 1] = 1
        W[k * n + 1, 1] = 1
        
    # Decomposition for j=1 (ranks 2, 3)
    # r=2: i=0, j=1
    U[0 * m + 1, 2] = 1
    for k in range(p):
        V[1 * p + k, 2] = 1
        W[k * n + 0, 2] = 1
        
    # r=3: i=1, j=1
    U[1 * m + 1, 3] = 1
    for k in range(p):
        V[1 * p + k, 3] = 1
        W[k * n + 1, 3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
