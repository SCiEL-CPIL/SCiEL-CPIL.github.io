"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    # Rank = m = 5 * (number of 2x2 blocks) = 5 * (n//2) * (p//2) = 5 * 2 * 3 = 30
    rank = m * (n // 2) * (p // 2)
    
    U = [[0] * rank for _ in range(n * m)]  # shape (20, 30)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 30)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 30)
    
    col = 0
    for block_row in range(n // 2):  # 0, 1
        for block_col in range(p // 2):  # 0, 1, 2
            for j in range(m):  # 0, 1, 2, 3, 4
                # For this block and j, set up one rank-1 term
                for i_offset in range(2):
                    i = block_row * 2 + i_offset
                    U[i * m + j][col] = 1
                for k_offset in range(2):
                    k = block_col * 2 + k_offset
                    V[j * p + k][col] = 1
                for i_offset in range(2):
                    for k_offset in range(2):
                        i = block_row * 2 + i_offset
                        k = block_col * 2 + k_offset
                        W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
