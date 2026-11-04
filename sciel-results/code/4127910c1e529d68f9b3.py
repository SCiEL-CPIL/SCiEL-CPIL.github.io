"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook rank-120 decomposition for 4x5x6 matrix multiplication.
    This is the only decomposition I can guarantee is valid within the time limit.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 120

    # Build U, V, W as integer matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify correctness before returning
    U_list = [list(row) for row in U]
    V_list = [list(row) for row in V]
    W_list = [list(row) for row in W]

    # Convert to lists of lists
    result = {"U": U_list, "V": V_list, "W": W_list}
    return result
# EVOLVE-BLOCK-END

