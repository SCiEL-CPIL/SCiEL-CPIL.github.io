"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p  # 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is all 1s
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                U[i * m + j][r] = 1
                
    # V[j*p+k, r] = 1 if r == k
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
            
    # W[k*n+i, r] = 1 if r == k
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
