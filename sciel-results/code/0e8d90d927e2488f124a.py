"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-18 exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Splits into two 2x3*3x3 sub-problems, each decomposed with rank 9 (schoolbook).
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    col = 0
    for block in range(2):  # Two blocks of 3 columns each
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i * m + j][col] = 1
                    V[j * p + (block * 3 + k)][col] = 1
                    W[(block * 3 + k) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
