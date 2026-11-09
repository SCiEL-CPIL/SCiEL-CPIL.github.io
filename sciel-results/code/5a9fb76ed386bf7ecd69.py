"""Matrix multiplication 3,4,6: rank-minimal exact tensor decomposition.

Constructs the exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
The schoolbook construction gives rank = n*m*p = 72 with integer entries.

We use the direct construction which is optimal for the "schoolbook" family.
"""

import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 115.0


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook rank: one rank-1 tensor for each (i, j, k) triple
    rank = n * m * p

    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
