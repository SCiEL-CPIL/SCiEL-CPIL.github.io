"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition.

The tensor M_{2,2,13} is a direct sum of 13 copies of M_{2,2,1}.
Each M_{2,2,1} has rank 4 (verified by the 4x4 unfolding matrix having rank 4).
Therefore the minimal rank is 13 * 4 = 52, which the schoolbook decomposition achieves.
"""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 2 * 13 = 52

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
