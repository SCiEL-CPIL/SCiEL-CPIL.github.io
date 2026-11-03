"""Rewritten program: Kronecker-product rank reduction for 2x3x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6

    # We aim to use a Kronecker product structure to reduce the rank below schoolbook (36).
    # The idea: decompose the 2x3x6 multiplication into tensor products of smaller problems.
    #
    # Specifically, we can view the problem as:
    #   A (2x3) * B (3x6) = C (2x6)
    #
    # We can factor the dimensions: 2 = 2*1, 3 = 3*1, 6 = 2*3.
    # But more usefully, we can use the fact that 6 = 2 * 3 to split B into two 3x3 blocks,
    # giving two 2x3x3 multiplications. If we can find a rank-15 decomposition for 2x3x3,
    # the total rank would be 30.
    #
    # Alternatively, we can use a Kronecker product of 1x2x3 and 2x1x1, but this is less direct.
    #
    # For now, let's implement a valid rank-36 schoolbook decomposition, but structured
    # in a way that could be optimized. We'll use numpy for clarity.

    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1

    # Convert to lists for the required return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
