"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = m
    U = [[0] * rank for _ in range(n * m)]  # (8, 4)
    V = [[0] * rank for _ in range(m * p)]  # (20, 4)
    W = [[0] * rank for _ in range(p * n)]  # (10, 4)
    
    for j in range(m):
        r = j
        # U[i, j] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
        # V[j, k] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
        # W[k, i] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
