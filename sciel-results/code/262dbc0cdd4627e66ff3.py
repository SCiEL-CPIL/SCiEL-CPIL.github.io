"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 4 decomposition for M_{2,2,12}
    # Based on M_{2,2,1} rank 4, extended to all k.
    
    # r=0: i=0, j=0
    U[0][0] = 1 # A[0,0]
    for k in range(p):
        V[0 * p + k][0] = 1 # B[0,k]
        W[k * n + 0][0] = 1 # C[0,k]
        
    # r=1: i=0, j=1
    U[1][1] = 1 # A[0,1]
    for k in range(p):
        V[1 * p + k][1] = 1 # B[1,k]
        W[k * n + 0][1] = 1 # C[0,k]
        
    # r=2: i=1, j=0
    U[2][2] = 1 # A[1,0]
    for k in range(p):
        V[0 * p + k][2] = 1 # B[0,k]
        W[k * n + 1][2] = 1 # C[1,k]
        
    # r=3: i=1, j=1
    U[3][3] = 1 # A[1,1]
    for k in range(p):
        V[1 * p + k][3] = 1 # B[1,k]
        W[k * n + 1][3] = 1 # C[1,k]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
