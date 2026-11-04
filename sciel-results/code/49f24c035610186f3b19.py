"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(m):
        for i in range(n):
            U[i * m + s][s] = 1
        for k in range(p):
            V[s * p + k][s] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][s] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
