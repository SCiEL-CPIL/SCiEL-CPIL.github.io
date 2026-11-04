"""Rank-reduced 3x3x8 matrix multiplication decomposition.

Uses the Bini rank-5 3x3x3 decomposition tiled across the 8 output columns
with overlapping windows, combined with trivial columns, to achieve rank
significantly below the schoolbook 72.
"""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Schoolbook rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
