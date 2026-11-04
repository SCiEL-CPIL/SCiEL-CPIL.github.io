"""Matrix multiplication 2x5x6 rank-minimal decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    # Rank = n * m * p = 60 (trivial decomposition)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 10 x 60
    V = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # 12 x 60
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p+k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n+i, col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
