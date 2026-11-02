"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build index arrays for the three tensor legs
    idx_U = np.arange(n * m)
    idx_V = np.arange(m * p)
    idx_W = np.arange(p * n)

    # For each rank column t = i*m + j + (k)*(m), assign the basis pattern
    # U[i*m + j][t] = 1, V[j*p + k][t] = 1, W[k*n + i][t] = 1
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
