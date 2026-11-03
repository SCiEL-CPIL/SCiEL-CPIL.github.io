import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # Minimum rank is m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i, j, r] = delta(j, r)
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if j == r:
                    U[i * m + j, r] = 1.0
                    
    # V[j, k, r] = delta(j, r)
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if j == r:
                    V[j * p + k, r] = 1.0
                    
    # W[k, i, r] = 1
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
