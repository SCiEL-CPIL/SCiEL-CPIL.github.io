"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 6 decomposition
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = delta(k, r)
    # W[k*n+i, r] = delta(k, r)
    rank = p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1
        for j in range(m):
            V[j * p + r][r] = 1
        for i in range(n):
            W[r * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
