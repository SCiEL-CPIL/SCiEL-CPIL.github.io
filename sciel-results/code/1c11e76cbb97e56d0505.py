"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    # U shape: (n*m, rank) = (9, 3)
    # V shape: (m*p, rank) = (12, 3)
    # W shape: (p*n, rank) = (12, 3)
    
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U: U[i*m + j, r] = 1 if r == j else 0
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if r == j:
                    U[i * m + j][r] = 1
    
    # Fill V: V[j*p + k, r] = 1 if r == j else 0
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if r == j:
                    V[j * p + k][r] = 1
    
    # Fill W: W[k*n + i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
