"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(p):
        for t in range(m):
            # U[i*m + j, r] = 1 when j == t
            for i in range(n):
                U[i * m + t][r] = 1
            # V[j*p + k, r] = 1 when j == t
            V[t * p + k][r] = 1
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
