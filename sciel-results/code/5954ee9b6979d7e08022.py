"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for s in range(2):
            r = k * 2 + s
            for i in range(n):
                for j in range(m):
                    # U[i*m + j, r] = delta[i, s]
                    U[i * m + j][r] = 1 if i == s else 0
                    # V[j*p + k, r] = 1
                    V[j * p + k][r] = 1
                    # W[k*n + i, r] = delta[i, s]
                    W[k * n + i][r] = 1 if i == s else 0
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
