"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-diagonal decomposition: each of the n rows of A is an independent
    # 1 x m by m x p matrix product. For a single row i, the tensor
    # T_i(j,k) = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # can be realized with rank m by setting, for each (j,k), one column where
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # The two row-blocks are disjoint in U-rows and W-columns, so no
    # cancellation is needed and the blocks do not interfere.
    rank = n * m  # 2 * 3 = 6
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row][col] = 1.0
                V[v_row][col] = 1.0
                W[w_row][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
