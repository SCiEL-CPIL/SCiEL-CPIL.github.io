"""Matrix multiplication 3,3,8 — rank-24 decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 3 * 8 = 24

    U = [[0] * rank for _ in range(n * m)]  # shape (9, 24)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 24)

    for k in range(p):
        for r in range(m):
            col = k * m + r
            # U[i*m + j, col] = 1 if j == r, for all i
            for i in range(n):
                U[i * m + r][col] = 1
            # V[j*p + k', col] = 1 if j == r and k' == k
            V[r * p + k][col] = 1
            # W[k'*n + i, col] = 1 if k' == k, for all i
            for i in range(n):
                W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
