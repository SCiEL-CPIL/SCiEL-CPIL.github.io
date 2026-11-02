"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p
    U_mat = np.zeros((total, n * m), dtype=float)
    V_mat = np.zeros((total, m * p), dtype=float)
    W_mat = np.zeros((total, p * n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[idx, i * m + j] = 1.0
                V_mat[idx, j * p + k] = 1.0
                W_mat[idx, k * n + i] = 1.0
                idx += 1
    return {
        "U": U_mat.T.tolist(),
        "V": V_mat.T.tolist(),
        "W": W_mat.T.tolist()
    }
# EVOLVE-BLOCK-END
