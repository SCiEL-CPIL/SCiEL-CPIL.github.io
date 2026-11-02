"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-57 decomposition for 3x3 times 3x7 matrix multiplication.
    Split into two 3x3x3 blocks (schoolbook rank 27 each) + one 3x3x1 block (rank 3).
    Total rank = 27 + 27 + 3 = 57.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    rank_total = 27 + 27 + 3  # 57
    
    U = [[0] * rank_total for _ in range(n * m)]  # 9 x 57
    V = [[0] * rank_total for _ in range(m * p)]  # 21 x 57
    W = [[0] * rank_total for _ in range(p * n)]  # 21 x 57
    
    col = 0
    # First 3 columns of B (k=0,1,2): schoolbook 3x3x3
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Next 3 columns of B (k=3,4,5): schoolbook 3x3x3
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
    
    # Last column of B (k=6): schoolbook 3x3x1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
