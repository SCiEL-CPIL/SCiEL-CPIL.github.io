"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen factors for 2x2x2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # We split the 4x4x6 problem into two 4x4x3 problems.
    # Each 4x4x3 problem is solved with rank 28 (using block Strassen for 4x4x4, 
    # effectively ignoring the 4th column, or using a specific 4x4x3 decomposition).
    # Actually, rank 28 for 4x4x4 is safe.
    # Total rank = 2 * 28 = 56.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 56
    
    # Helper to build rank 28 factors for 4x4x4
    def build_4x4x4_factors(offset):
        U_local = [[0] * 28 for _ in range(16)]
        V_local = [[0] * 28 for _ in range(16)]
        W_local = [[0] * 28 for _ in range(16)]
        
        # 4 blocks of output, each 2x2.
        # For each output block (bi, bj), we use 7 Strassen products.
        # The inputs are specific 2x2 blocks of A and B.
        
        for bi in range(2):
            for bj in range(2):
                # Offset for this output block's products
                prod_offset = (bi * 2 + bj) * 7
                
                # Input A blocks: rows bi*2 to bi*2+2, cols 0 to 4
                # Input B blocks: rows 0 to 4, cols bj*2 to bj*2+2
                
                # Strassen on A_block (2x4) and B_block (4x2)?
                # No, Strassen is for 2x2.
                # We can split A_block into two 2x2 blocks and B_block into two 2x2 blocks.
                # This is effectively doing 4 2x2x2 multiplications?
                # No, it's one 2x4x2 multiplication.
                
                # Let's just use the schoolbook for the 2x4x2 part?
                # Rank of 2x4x2 is 8.
                # 4 blocks * 8 = 32.
                
                # Let's use the rank 7 Strassen for each 2x2x2 sub-block of the 4x4x4.
                # There are 4 such sub-blocks in the input A and 4 in B.
                # This is not straightforward.
                
                # I will fall back to the seed program's rank 84 if I can't quickly construct rank 56.
                # But rank 56 is much better.
                
                pass
        
        return U_local, V_local, W_local
    
    # Given the complexity, I will return the seed program's result (rank 84) 
    # to ensure correctness, as I cannot fully derive the rank 56 factors in this thought process.
    
    # Re-implementing seed logic
    n_blocks = n // 2
    m_blocks = m // 2
    p_blocks = p // 2
    block_size = 2
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = bi * block_size + di
                            j = bj * block_size + dj
                            U[i * m + j][col] = S_U[di * block_size + dj][c]
                    
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = bj * block_size + dj
                            k = bk * block_size + dk
                            V[j * p + k][col] = S_V[dj * block_size + dk][c]
                    
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = bk * block_size + dk
                            i = bi * block_size + di
                            W[k * n + i][col] = S_W[dk * block_size + di][c]
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
