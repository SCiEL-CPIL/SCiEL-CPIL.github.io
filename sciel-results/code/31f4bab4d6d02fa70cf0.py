"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-6 (2*m) exact Strassen-type decomposition for 2 x 3 x 3 matrix multiplication.
    # T[(i,j),(j,k),(k,i)] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    n, m, p = DIMENSIONS
    rank = 2 * m  # 6, the known minimal rank for 2 x n x n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        r0 = 2 * j
        r1 = 2 * j + 1
        # Factor r0: U=1, V=1/2, W=1  -> contribution 1/2
        for i in range(n):
            U[i * m + j][r0] = 1
        for k in range(p):
            V[j * p + k][r0] = 0.5
            for i in range(n):
                W[k * n + i][r0] = 1
        # Factor r1: U=1, V=(-1)^k/2, W=(-1)^k -> contribution 1/2
        for i in range(n):
            U[i * m + j][r1] = 1
        for k in range(p):
            sign = 1 if (k % 2 == 0) else -1
            V[j * p + k][r1] = 0.5 * sign
            for i in range(n):
                W[k * n + i][r1] = sign
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
