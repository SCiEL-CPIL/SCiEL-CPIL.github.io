"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 for 4x4x4 (7 Strassen products * 8 schoolbook terms)
    # Rank 16 for 4x4x1 (Schoolbook)
    # Rank 16 for 4x4x1 (Schoolbook)
    # Total Rank = 56 + 16 + 16 = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a rank-1 tensor
    def add_tensor(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col_offset
        if col_offset >= rank:
            return
        U[u_idx][col_offset] = u_val
        V[v_idx][col_offset] = v_val
        W[w_idx][col_offset] = w_val
        col_offset += 1

    # Part 1: 4x4x4 using 7 Strassen products, each with 8 schoolbook terms
    # Strassen Products:
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    # We iterate through the 7 products.
    # For each product, we have a combination of A blocks and B blocks.
    # We expand each 2x2 product into 8 scalar products.
    
    # Define the 7 products' A and B block combinations
    # A_blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B_blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    products = [
        # (A_comb, B_comb, C_comb)
        # A_comb is a list of (block_idx, coeff)
        # B_comb is a list of (block_idx, coeff)
        # C_comb is a list of (block_idx, coeff)
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for a_comb, b_comb, c_comb in products:
        # Expand the 2x2 product
        # A_part = sum(a_coeff * A_block)
        # B_part = sum(b_coeff * B_block)
        # C_part = sum(c_coeff * C_block)
        
        # The product is A_part * B_part.
        # Each A_block is 2x2, each B_block is 2x2.
        # We need to sum over all combinations of blocks.
        
        # For each term in A_comb and B_comb:
        for (a_blk, a_coeff) in a_comb:
            for (b_blk, b_coeff) in b_comb:
                # We have a product of A_block a_blk and B_block b_blk.
                # This is a 2x2 matrix multiplication.
                # We use schoolbook for this 2x2 multiplication (8 terms).
                
                # A_block a_blk indices:
                # a_blk 0 (A00): rows 0,1 cols 0,1
                # a_blk 1 (A01): rows 0,1 cols 2,3
                # a_blk 2 (A10): rows 2,3 cols 0,1
                # a_blk 3 (A11): rows 2,3 cols 2,3
                
                # B_block b_blk indices:
                # b_blk 0 (B00): rows 0,1 cols 0,1
                # b_blk 1 (B01): rows 0,1 cols 2,3
                # b_blk 2 (B10): rows 2,3 cols 0,1
                # b_blk 3 (B11): rows 2,3 cols 2,3
                
                # The product P = A_block * B_block is 2x2.
                # P[i,j] = sum_k A_block[i,k] * B_block[k,j]
                
                # We add this to C_blocks according to c_comb.
                
                for (c_blk, c_coeff) in c_comb:
                    # C_block c_blk indices:
                    # c_blk 0 (C00): rows 0,1 cols 0,1
                    # c_blk 1 (C01): rows 0,1 cols 2,3
                    # c_blk 2 (C10): rows 2,3 cols 0,1
                    # c_blk 3 (C11): rows 2,3 cols 2,3
                    
                    # For each of the 8 terms in the 2x2 multiplication:
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                # A index: (a_blk_row + i, a_blk_col + k)
                                a_row = (a_blk // 2) * 2 + i
                                a_col = (a_blk % 2) * 2 + k
                                a_idx = a_row * 4 + a_col
                                
                                # B index: (b_blk_row + k, b_blk_col + j)
                                b_row = (b_blk // 2) * 2 + k
                                b_col = (b_blk % 2) * 2 + j
                                b_idx = b_row * 6 + b_col # m=4, p=6, so B is 4x6. Index is row*6+col.
                                # Wait, B is 4x6. Blocks are 2x2.
                                # B00: rows 0,1 cols 0,1
                                # B01: rows 0,1 cols 2,3
                                # B10: rows 2,3 cols 0,1
                                # B11: rows 2,3 cols 2,3
                                # This mapping is correct for the first 4 columns.
                                
                                # C index: (c_blk_row + i, c_blk_col + j)
                                c_row = (c_blk // 2) * 2 + i
                                c_col = (c_blk % 2) * 2 + j
                                c_idx = c_row * 4 + c_col # p=6, n=4, so C is 6x4? No, C is 4x6.
                                # W is p x n. W[k*n+i, r].
                                # C[i,j] corresponds to W[j*n+i, r]?
                                # Task: W[k*n+i, r].
                                # C is 4x6. i is row (0-3), j is col (0-5).
                                # W index: j*4 + i.
                                c_idx_w = c_col * 4 + c_row
                                
                                coeff = a_coeff * b_coeff * c_coeff
                                if coeff != 0:
                                    add_tensor(a_idx, coeff, b_idx, 1, c_idx_w, 1)
                                    # Wait, V and W should have 1?
                                    # U[a_idx] = coeff
                                    # V[b_idx] = 1
                                    # W[c_idx_w] = 1
                                    # This is correct for rank-1 tensor u v w.
    
    # Part 2: 4x4x1 for column 4
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = j * 6 + 4
            c_idx_w = 4 * 4 + i # col 4, row i
            add_tensor(a_idx, 1, b_idx, 1, c_idx_w, 1)
            
    # Part 3: 4x4x1 for column 5
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = j * 6 + 5
            c_idx_w = 5 * 4 + i # col 5, row i
            add_tensor(a_idx, 1, b_idx, 1, c_idx_w, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
