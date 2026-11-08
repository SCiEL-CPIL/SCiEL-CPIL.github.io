"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = n * m * p  # 36

    U = [[0] * rank for _ in range(n * m)]   # 6 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 12 x 36

    # Each rank-1 term corresponds to a scalar multiplication A[i,j] * B[j,k] -> C[i,k]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i  (official trace convention W[k*n + i, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
