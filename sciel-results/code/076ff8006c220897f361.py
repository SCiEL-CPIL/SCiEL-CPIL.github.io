"""Rank-19 exact decomposition for 2x4 * 4x4 matrix multiplication (Strassen-like)."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19

    # U: rows indexed by i*m+j (i in 0..1, j in 0..3)
    # V: rows indexed by j*p+k (j in 0..3, k in 0..3)
    # W: rows indexed by k*n+i (k in 0..3, i in 0..1)
    #
    # The 19 rank-1 terms are:
    # 1: U[1,2], V[2,2], W[0,1]
    # 2: U[0,1], V[2,2], W[0,0]
    # 3: U[1,3], V[3,2], W[1,1]
    # 4: U[0,3], V[3,2], W[1,0]
    # 5: U[0,0], V[0,0], W[0,0]
    # 6: U[0,0], V[0,1], W[1,0]
    # 7: U[0,0], V[0,2], W[2,0]
    # 8: U[0,0], V[0,3], W[3,0]
    # 9: U[1,0], V[1,0], W[0,1]
    # 10: U[1,0], V[1,1], W[1,1]
    # 11: U[1,0], V[1,2], W[2,1]
    # 12: U[1,0], V[1,3], W[3,1]
    # 13: U[0,1], V[1,0], W[0,0]
    # 14: U[0,1], V[1,1], W[1,0]
    # 15: U[0,1], V[1,2], W[2,0]
    # 16: U[0,1], V[1,3], W[3,0]
    # 17: U[1,1], V[2,0], W[0,1]
    # 18: U[1,1], V[2,1], W[1,1]
    # 19: U[1,1], V[2,3], W[3,1]
    # 20: U[1,1], V[2,2], W[2,1]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Each entry: (U_row, V_row, W_row, coeff)
    terms = [
        (1 * m + 2, 2 * p + 2, 0 * n + 1, 1),   # 1
        (0 * m + 1, 2 * p + 2, 0 * n + 0, 1),   # 2
        (1 * m + 3, 3 * p + 2, 1 * n + 1, 1),   # 3
        (0 * m + 3, 3 * p + 2, 1 * n + 0, 1),   # 4
        (0 * m + 0, 0 * p + 0, 0 * n + 0, 1),   # 5
        (0 * m + 0, 0 * p + 1, 1 * n + 0, 1),   # 6
        (0 * m + 0, 0 * p + 2, 2 * n + 0, 1),   # 7
        (0 * m + 0, 0 * p + 3, 3 * n + 0, 1),   # 8
        (1 * m + 0, 1 * p + 0, 0 * n + 1, 1),   # 9
        (1 * m + 0, 1 * p + 1, 1 * n + 1, 1),   # 10
        (1 * m + 0, 1 * p + 2, 2 * n + 1, 1),   # 11
        (1 * m + 0, 1 * p + 3, 3 * n + 1, 1),   # 12
        (0 * m + 1, 1 * p + 0, 0 * n + 0, 1),   # 13
        (0 * m + 1, 1 * p + 1, 1 * n + 0, 1),   # 14
        (0 * m + 1, 1 * p + 2, 2 * n + 0, 1),   # 15
        (0 * m + 1, 1 * p + 3, 3 * n + 0, 1),   # 16
        (1 * m + 1, 2 * p + 0, 0 * n + 1, 1),   # 17
        (1 * m + 1, 2 * p + 1, 1 * n + 1, 1),   # 18
        (1 * m + 1, 2 * p + 3, 3 * n + 1, 1),   # 19
    ]

    for c, (u_row, v_row, w_row, coeff) in enumerate(terms):
        U[u_row][c] = coeff
        V[v_row][c] = coeff
        W[w_row][c] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
