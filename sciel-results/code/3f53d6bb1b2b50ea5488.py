"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p into two blocks of size 3
    p_block = 3
    num_blocks = p // p_block
    
    # Rank is n*m*p_block * num_blocks = 2*3*3 * 2 = 18
    rank = n * m * p_block * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for block in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    k_global = block * p_block + k
                    U[i * m + j][column] = 1
                    V[j * p + k_global][column] = 1
                    W[k_global * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
