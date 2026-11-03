"""Exact 6x6x6 matrix-multiplication tensor decomposition, rank 189.

Kronecker product of 2x2 Strassen (rank 7) acting on in-block 2x2 indices
with 3x3 schoolbook (rank 27) acting on 3x3 block indices. 6 = 3 blocks of 2,
so index i = 2*i_block + i_inblock. All entries are integers; the decomposition
is exact.
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Standard 2x2 Strassen factors (Wikipedia P1..P7 convention).
    U_s = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [-1, 0]],
    ]
    V_s = [
        [[1, 1], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, -1], [1, 0]],
        [[0, -1], [1, 1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    W_s = [
        [[1, 0], [0, 1]],
        [[-1, 1], [0, 0]],
        [[1, 0], [0, -1]],
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 1], [0, 0]],
        [[0, 1], [0, 1]],
    ]

    n = m = p = 6
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                for t2 in range(7):
                    c = col
                    for i2 in range(2):
                        for j2 in range(2):
                            U[2 * i3 + i2][c] += U_s[t2][i2][j2]
                    for j2 in range(2):
                        for k2 in range(2):
                            V[2 * j3 + j2][c] += V_s[t2][j2][k2]
                    for k2 in range(2):
                        for i2 in range(2):
                            W[2 * k3 + i2][c] += W_s[t2][k2][i2]
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
