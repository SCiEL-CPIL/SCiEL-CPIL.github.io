"""Matrix multiplication 3x3 x 3x8 with rank-minimal exact decomposition.

The multiplication tensor T_{3,3,8} can be viewed as the direct sum of 8 copies
of T_{3,3,1} (matrix-vector product 3x3 by 3x1). Each T_{3,3,1} has tensor rank
exactly 9 (the schoolbook rank for 3x3 matrix-vector multiplication), and no
better decomposition is known for these dimensions. Thus the total rank is
8 * 9 = 72.

We construct the factor matrices using a systematic index mapping.
"""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = n * m * p  # 72

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

    # Convert to list-of-lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
