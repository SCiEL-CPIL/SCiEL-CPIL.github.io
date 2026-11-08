"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = min(n, m, p)  # rank 3
    # U shape: (n*m, rank), U[i*m + j, r] = 1 if i == r else 0
    U = [[0] * rank for _ in range(n * m)]
    # V shape: (m*p, rank), V[j*p + k, r] = 1 for all
    V = [[1] * rank for _ in range(m * p)]
    # W shape: (p*n, rank), W[k*n + i, r] = 1 for all
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1  # only column i (which equals r when r==i) is 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
