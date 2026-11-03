"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (2, 3, 9)

    rank = n * m * p  # 54, exact schoolbook (verified exact)
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    flat = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Factor flat = (i*m+j)*p + k
                U[i * m + j, flat] = 1
                V[j * p + k, flat] = 1
                W[k * n + i, flat] = 1
                flat += 1

    return {
        "U": [row.tolist() for row in U],
        "V": [row.tolist() for row in V],
        "W": [row.tolist() for row in W],
    }
# EVOLVE-BLOCK-END
