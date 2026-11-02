"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Set U[i*m + 0][0] = 1 for all i
    for i in range(n):
        U[i * m][0] = 1
        
    # Set V[0*p + k][0] = 1 for all k
    for k in range(p):
        V[k][0] = 1
        
    # Set W[k*n + i][0] = 1 for all k, i
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
