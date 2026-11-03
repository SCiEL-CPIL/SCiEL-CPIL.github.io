import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use the schoolbook rank: n*m*p
    rank = n * m * p

    # We build U, V, W as dense integer matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # For each (i, j, k) we create one rank-1 term:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Return as nested lists of integers (exact)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
