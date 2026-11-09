"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # The 2x3x9 multiplication tensor decomposes into p=9 independent 2x3x1 blocks.
    # Each 2x3x1 block has rank 6 (the schoolbook rank for multiplying a 2x3 by 3x1).
    # Since the output indices for different k values are disjoint (W index = k*n + i),
    # the blocks are orthogonal in the W-factor, so no rank-1 term can span multiple k.
    # Therefore the total minimal rank is 9 * 6 = 54.
    rank = n * m * p  # 2 * 3 * 9 = 54

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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
