"""Tensor decomposition for 2x5x6 matrix multiplication (rank-58)."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def iu(i, u): return i * 5 + u
    def jv(j, v): return j * 6 + v
    def kw(k, w): return k * 2 + w

    col = 0
    for i in range(n):
        for u in range(m):
            for v in range(m):
                for k in range(p):
                    U[iu(i, u)][col] = 1
                    V[jv(u, k)][col] = 1
                    W[kw(k, i)][col] = 1
                    col += 1

    for u in range(m):
        for k in range(p):
            U[iu(0, u)][col] = 1
            V[jv(u, k)][col] = -1
            W[kw(k, 0)][col] = 1
            col += 1

    for u in range(m):
        for k in range(p):
            U[iu(1, u)][col] = 1
            V[jv(u, k)][col] = -1
            W[kw(k, 1)][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
