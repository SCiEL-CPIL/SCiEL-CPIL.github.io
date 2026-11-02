"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 8 x 12

    # Each term r covers: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We group by j and k: for each (j,k) pair, one term handles both i values.
    # There are m*p = 12 such (j,k) pairs, so rank = 12.

    r = 0
    for j in range(m):
        for k in range(p):
            # U entries: for i in {0,1}, U[i*m + j, r] = 1
            U[0 * m + j][r] = 1  # i=0
            U[1 * m + j][r] = 1  # i=1
            # V entry: V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # W entries: for i in {0,1}, W[k*n + i, r] = 1
            W[k * n + 0][r] = 1  # i=0
            W[k * n + 1][r] = 1  # i=1
            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
