"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            # U: for all i, set U[i*m + j0, c] = 1
            for i in range(n):
                U[i * m + j0][c] = 1
            # V: set V[j0*p + k0, c] = 1
            V[j0 * p + k0][c] = 1
            # W: for all k, i, set W[k*n + i, c] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
