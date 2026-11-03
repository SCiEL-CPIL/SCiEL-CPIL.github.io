"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]  # 6 x 3
    V = [[0] * rank for _ in range(m * p)]  # 21 x 3
    W = [[0] * rank for _ in range(p * n)]  # 14 x 3
    
    for b in range(m):
        for a in range(n):
            U[a * m + b][b] = 1
        for d in range(p):
            V[b * p + d][b] = 1
        for a in range(n):
            for d in range(p):
                W[a * n + d][b] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
