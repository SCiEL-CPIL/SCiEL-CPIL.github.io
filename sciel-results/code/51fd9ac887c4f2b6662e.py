"""Matrix multiplication 2,3,9 — rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54

    # Build factor matrices with the standard rank-54 decomposition.
    # U[i*m + j, r] = 1 iff r == (i*m + j)*p + k
    # V[j*p + k, r] = 1 iff r == (i*m + j)*p + k  (same r index)
    # W[k*n + i, r] = 1 iff r == (i*m + j)*p + k
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
# EVOLVE-BLOCK-END
