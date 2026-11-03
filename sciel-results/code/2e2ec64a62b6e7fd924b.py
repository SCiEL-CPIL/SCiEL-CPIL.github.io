"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m * p / p? No.
    # We split the p dimension into p blocks of size 1.
    # Each block is a 2 x 3 x 1 multiplication.
    # Rank of 2 x 3 x 1 is 3 (dot product of length 3).
    # Total rank = p * 3 = 6 * 3 = 18.
    rank = p * m  # 6 * 3 = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for j in range(m):
            r = k * m + j
            for i in range(n):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
