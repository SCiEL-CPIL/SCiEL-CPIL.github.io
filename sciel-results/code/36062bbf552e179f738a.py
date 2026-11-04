"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i0 in range(n):
        for k0 in range(p):
            col = i0 * p + k0
            # U[i*m+j, col] = 1 if i == i0
            for j in range(m):
                U[i0 * m + j][col] = 1
            
            # V[j*p+k, col] = 1 for all j, k
            for j in range(m):
                for k in range(p):
                    V[j * p + k][col] = 1
            
            # W[k*n+i, col] = 1 if k == k0
            for k in range(p):
                for i in range(n):
                    if k == k0:
                        W[k * n + i][col] = 1
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
