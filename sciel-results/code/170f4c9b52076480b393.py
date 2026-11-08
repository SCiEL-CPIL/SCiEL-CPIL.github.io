"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # Direct block decomposition: split B into 3 blocks of 2 columns
    # Each block: 2x3 * 3x2 -> 2x2, rank 6 (trivial)
    # Total rank = 3 * 6 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    for b in range(3):
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    col = b * 6 + i * 3 + j * 2 + k
                    U[i * 3 + j][col] = 1
                    V[j * p + (2 * b + k)][col] = 1
                    W[(2 * b + k) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
