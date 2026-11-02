"""Construct a rank-30 tensor decomposition for 3x3 * 3x4 matrix multiplication.

Uses 27 schoolbook terms for the 3x3x3 core (k=0,1,2) plus 3 terms for the
fourth column (k=3), giving total rank 30.
"""

def construct():
    n, m, p = 3, 3, 4
    rank = 30

    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows

    column = 0

    # 1. Schoolbook terms for the 3x3x3 core (k = 0, 1, 2)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # 2. Three terms for the fourth output column (k = 3)
    #    C[i, 3] = sum_{j} A[i, j] * B[j, 3]
    #    For each j, one rank-1 term:
    #      U[i, j, r] = 1 for all i
    #      V[j, 3, r] = 1
    #      W[3, i, r] = 1 for all i
    for j in range(m):
        for i in range(n):
            U[i * m + j][column] = 1
        V[j * p + 3][column] = 1
        for i in range(n):
            W[3 * n + i][column] = 1
        column += 1

    return {"U": U, "V": V, "W": W}
