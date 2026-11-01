"""Matrix multiplication 2x2 * 2x11 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 2x2 * 2x11, the standard bilinear decomposition has rank n*m*p = 44
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j]*B[j,k] contributes to C[i,k]
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
