"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 4 decomposition
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: i=0, j=0
    U[0 * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        W[k * n + 0][0] = 1
        
    # r=1: i=0, j=1
    U[0 * m + 1][1] = 1
    for k in range(p):
        V[1 * p + k][1] = 1
        W[k * n + 0][1] = 1
        
    # r=2: i=1, j=0
    U[1 * m + 0][2] = 1
    for k in range(p):
        V[0 * p + k][2] = 1
        W[k * n + 1][2] = 1
        
    # r=3: i=1, j=1
    U[1 * m + 1][3] = 1
    for k in range(p):
        V[1 * p + k][3] = 1
        W[k * n + 1][3] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
