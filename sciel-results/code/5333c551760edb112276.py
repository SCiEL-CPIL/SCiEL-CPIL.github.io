"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # 5
    # U shape: (n*m, rank) = (10, 5)
    # U[i*m+j, c] = 1 if c == j else 0
    U = [[0] * rank for _ in range(n * m)]
    # V shape: (m*p, rank) = (30, 5)
    # V[j*p+k, c] = 1 if c == j else 0
    V = [[0] * rank for _ in range(m * p)]
    # W shape: (p*n, rank) = (12, 5)
    # W[k*n+i, c] = 1 for all k, i, c
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
