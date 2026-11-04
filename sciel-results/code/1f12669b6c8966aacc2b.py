DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i0 in range(n):
        for j0 in range(m):
            U[i0 * m + j0][r] = 1
            for k in range(p):
                V[j0 * p + k][r] = 1
            for k in range(p):
                W[k * n + i0][r] = 1
            r += 1
    return {"U": U, "V": V, "W": W}
