DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Block 1: use rank-27 schoolbook for first 3 columns of B (k = 0,1,2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    # Block 2: use rank-9 schoolbook for remaining 4 columns of B (k = 3,4,5,6)
    # We only need 4 output rows in W, but W dimension is p*n = 21.
    # For these columns, only W[3 + i][col] is used for i = 0,1,2.
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
