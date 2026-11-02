"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank is determined by m (or p, or n*m/p etc depending on symmetry)
    # Here we use rank = m = 5
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                # U[i*m + j, r] is 1 if j == r
                if j == r:
                    U[i * m + j][r] = 1
        
        for j in range(m):
            for k in range(p):
                # V[j*p + k, r] is 1 if j == r
                if j == r:
                    V[j * p + k][r] = 1
        
        for k in range(p):
            for i in range(n):
                # W[k*n + i, r] is 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
