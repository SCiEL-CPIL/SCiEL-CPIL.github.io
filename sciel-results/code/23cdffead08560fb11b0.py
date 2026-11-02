"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank_val = n * m * p
    U = [[0] * rank_val for _ in range(n * m)]
    V = [[0] * rank_val for _ in range(m * p)]
    W = [[0] * rank_val for _ in range(p * n)]
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
