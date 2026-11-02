import numpy as np

def construct():
    n, m, p = 2, 3, 9
    # Each output column k is a matrix-vector product A (2x3) * B[:,k] (3x1).
    # The (2,3,1) tensor has rank 6 (schoolbook), and the 9 columns are independent
    # in the output, so the total rank is 9 * 6 = 54.
    # We build the factor matrices directly via nested loops, one rank slot per
    # (i, j, k) triple, matching the trace convention W[k*n + i, r].
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}
