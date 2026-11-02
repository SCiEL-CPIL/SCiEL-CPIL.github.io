"""Rewritten program for matrix multiplication 4x5 by 5x5 (tensor decomposition)."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal (for this implementation) exact tensor decomposition.
    # For 4x5 * 5x5, the schoolbook rank is 4*5*5 = 100.
    # We return a valid decomposition with integer entries (all 1s on the diagonal tensor).
    # The evaluator requires exact decomposition with integer/half-integer entries.
    n, m, p = DIMENSIONS
    rank = n * m * p
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
