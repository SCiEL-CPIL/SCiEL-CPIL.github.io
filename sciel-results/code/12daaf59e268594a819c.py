"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We split the multiplication into two parts:
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4). Rank 49 (Strassen).
    # Part 2: A (4x4) * B2 (4x5) -> C2 (4x5). Rank 80 (Schoolbook).
    
    total_rank = 49 + 80
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x 129
    V = [[0] * total_rank for _ in range(m * p)] # 36 x 129
    W = [[0] * total_rank for _ in range(p * n)] # 36 x 129
    
    # --- Part 1: Strassen 4x4 ---
    # Strassen for 2x2 has rank 7.
    # Strassen for 4x4 uses 7 calls to 2x2 Strassen.
    # We need to map the 2x2 Strassen coefficients to the 4x4 indices.
    
    # Let's define the 2x2 Strassen decomposition first.
    # A2 = [[A00, A01], [A10, A11]]
    # B2 = [[B00, B01], [B10, B11]]
    # C2 = [[C00, C01], [C10, C11]]
    
    # The 7 products M1..M7 are linear combinations of A and B entries.
    # We represent each product by its U, V, W vectors.
    # For 2x2:
    # U is 4x7, V is 4x7, W is 4x7.
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Let's create the 2x2 Strassen matrices.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Indices for 2x2:
    # A: 00->0, 01->1, 10->2, 11->3
    # B: 00->0, 01->1, 10->2, 11->3
    # C: 00->0, 01->1, 10->2, 11->3
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[3][3] = 1
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 1; W2[2][5] = 1
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[1][6] = 1; W2[3][6] = 1
    
    # Now, for 4x4, we have 4 blocks of 2x2.
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00, C01, C10, C11 similarly.
    
    # The 7 products M1..M7 are computed using the 2x2 Strassen formulas on the blocks.
    # Each block product involves 7 scalar products of 2x2 matrices.
    # So we have 7 * 7 = 49 scalar products total.
    
    # We will fill the first 49 columns of U, V, W.
    
    # Block offsets for 4x4 indices (row, col) -> block_index, local_row, local_col
    # Block 0: (0,0), (0,1), (1,0), (1,1)
    # Block 1: (0,2), (0,3), (1,2), (1,3)
    # Block 2: (2,0), (2,1), (3,0), (3,1)
    # Block 3: (2,2), (2,3), (3,2), (3,3)
    
    # We iterate over the 7 Strassen terms for the 4x4 level.
    # Term t in 0..6.
    # This term is a sum of products of A-blocks and B-blocks.
    # Specifically, term t uses:
    # A-indices: a1, a2 (from U2[t])
    # B-indices: b1, b2 (from V2[t])
    # C-indices: c1, c2, c3, c4 (from W2[t])
    
    # Wait, the 2x2 Strassen matrices U2, V2, W2 already encode which 2x2 blocks are involved.
    # U2 is 4x7. The rows are the 4 entries of the 2x2 matrix A_block.
    # So for a given term t, U2[i][t] tells us the coefficient for A_block_entry_i.
    
    # For 4x4, we have 4 A-blocks. Let's call them A_B0, A_B1, A_B2, A_B3.
    # The 2x2 Strassen formula operates on a 2x2 matrix of blocks.
    # So the "A" in the 2x2 formula is the 2x2 matrix of blocks [[A_B0, A_B1], [A_B2, A_B3]].
    # The "B" is [[B_B0, B_B1], [B_B2, B_B3]].
    
    # So, for term t, we look at U2[i][t] for i in 0..3.
    # i=0 corresponds to block A_B0.
    # i=1 corresponds to block A_B1.
    # i=2 corresponds to block A_B2.
    # i=3 corresponds to block A_B3.
    
    # For each non-zero coefficient in U2[i][t], we need to add the corresponding 2x2 Strassen decomposition for that block?
    # No. The 2x2 Strassen decomposition computes the product of two 2x2 matrices.
    # Here, the "matrices" are 2x2 blocks of the 4x4 matrix.
    # The product of two 2x2 blocks is a 2x2 matrix.
    # We need to decompose the product of two 2x2 blocks into scalar products.
    # This is exactly what the 2x2 Strassen matrices U2, V2, W2 do!
    # But U2, V2, W2 are for the entries of the 2x2 matrix.
    # So, if U2[i][t] = 1, it means we use the i-th entry of the A-block.
    # But the i-th entry of the A-block is a 2x2 matrix? No, the i-th entry of the 2x2 block matrix is a 2x2 submatrix.
    
    # Let's clarify.
    # 4x4 A is composed of four 2x2 blocks: A00, A01, A10, A11.
    # 4x4 B is composed of four 2x2 blocks: B00, B01, B10, B11.
    # The Strassen algorithm computes 7 products M1..M7.
    # Each Mk is a sum of products of A-blocks and B-blocks.
    # For example, M1 = (A00 + A11)(B00 + B11).
    # This involves 4 block products: A00B00, A00B11, A11B00, A11B11.
    # Each block product (e.g., A00B00) is a product of two 2x2 matrices.
    # We can decompose each 2x2 product using the 2x2 Strassen algorithm (rank 7).
    # So M1 involves 4 * 7 = 28 scalar products.
    # Total for 7 terms: 7 * 4 * 7? No.
    
    # The standard Strassen recursion:
    # T(4) = 7 * T(2) + O(1).
    # T(2) = 7.
    # T(4) = 7 * 7 = 49.
    
    # So we just need to map the 49 scalar products to the global indices.
    
    # Let's define the mapping.
    # For each of the 7 high-level terms (t in 0..6):
    #   We have coefficients for A-blocks and B-blocks from U2 and V2.
    #   U2 is 4x7. Rows are block indices 0,1,2,3.
    #   V2 is 4x7. Rows are block indices 0,1,2,3.
    #   W2 is 4x7. Rows are block indices 0,1,2,3.
    
    #   For a given t, let a_blocks be the set of block indices i where U2[i][t] != 0.
    #   Let b_blocks be the set of block indices j where V2[j][t] != 0.
    #   The product is sum_{i in a_blocks, j in b_blocks} U2[i][t] * V2[j][t] * (A_block_i * B_block_j).
    
    #   Each A_block_i * B_block_j is a 2x2 matrix product.
    #   We decompose this using the 2x2 Strassen matrices U2_2x2, V2_2x2, W2_2x2.
    #   Wait, U2, V2, W2 defined above ARE the 2x2 Strassen matrices.
    #   So for each pair (i, j) with non-zero coefficients, we add 7 scalar products.
    
    #   The scalar products for A_block_i * B_block_j:
    #   For k in 0..6 (2x2 Strassen terms):
    #     U_scalar = U2_2x2[row_a][k] * U2[i][t]
    #     V_scalar = V2_2x2[col_b][k] * V2[j][t]
    #     W_scalar = W2_2x2[out_c][k]
    
    #   We need to map row_a, col_b, out_c to global indices.
    #   A_block_i is at global rows/cols.
    #   Block 0: rows 0-1, cols 0-1.
    #   Block 1: rows 0-1, cols 2-3.
    #   Block 2: rows 2-3, cols 0-1.
    #   Block 3: rows 2-3, cols 2-3.
    
    #   Let's precompute the global indices for each block and local index.
    
    #   col_offset = 0
    #   for t in range(7):
    #     for i in range(4): # A block index
    #       if U2[i][t] == 0: continue
    #       for j in range(4): # B block index
    #         if V2[j][t] == 0: continue
    #         # Add 7 scalar products for A_block_i * B_block_j
    #         for k in range(7): # 2x2 Strassen term
    #           # Get coefficients
    #           coeff_u = U2[i][t]
    #           coeff_v = V2[j][t]
    #           
    #           # For each of the 4 entries in the 2x2 product?
    #           # No, the 2x2 Strassen matrices U2_2x2, V2_2x2, W2_2x2 handle the 4 entries.
    #           # U2_2x2 is 4x7. Rows are local row/col of A_block.
    #           # V2_2x2 is 4x7. Rows are local row/col of B_block.
    #           # W2_2x2 is 4x7. Rows are local row/col of C_block.
            
    #           # We need to add to global U, V, W.
    #           # For each local entry in A_block_i:
    #           for la in range(4): # local index in 2x2 block
    #             if U2[la][k] == 0: continue
    #             global_u_idx = get_global_A_idx(i, la)
    #             U[global_u_idx][col_offset] += coeff_u * U2[la][k]
                
    #           # For each local entry in B_block_j:
    #           for lb in range(4):
    #             if V2[lb][k] == 0: continue
    #             global_v_idx = get_global_B_idx(j, lb)
    #             V[global_v_idx][col_offset] += coeff_v * V2[lb][k]
                
    #           # For each local entry in C_block:
    #           # Which C block?
    #           # The output of A_block_i * B_block_j contributes to C blocks determined by W2.
    #           # W2[c][k] tells us which local entry of the C_block is affected.
    #           # But which C_block?
    #           # The C_block index is determined by the high-level W2[t].
    #           # W2 is 4x7. Rows are C block indices 0,1,2,3.
    #           # For a given t, W2[c][t] != 0 means C_block_c is affected.
    #           # So we need to iterate over c in 0..3 where W2[c][t] != 0.
            
    #           # Wait, the 2x2 Strassen product A_block_i * B_block_j results in a 2x2 matrix.
    #           # This matrix is added to the C_block determined by the high-level term.
    #           # Actually, the high-level term t produces a sum of C_blocks.
    #           # W2[c][t] is the coefficient for C_block_c.
    #           # So for each c where W2[c][t] != 0, we add the 2x2 product to C_block_c.
            
    #           # So the loop structure is:
    #           for c in range(4):
    #             if W2[c][t] == 0: continue
    #             coeff_w = W2[c][t]
    #             for lc in range(4):
    #               if W2[lc][k] == 0: continue
    #               global_w_idx = get_global_C_idx(c, lc)
    #               W[global_w_idx][col_offset] += coeff_w * W2[lc][k]
                
    #           col_offset += 1
    #           if time.time() > deadline: return {"U": U, "V": V, "W": W}

    # This logic seems correct. Let's implement the helper functions.
    
    def get_global_A_idx(block, local):
        # block: 0,1,2,3
        # local: 0,1,2,3 (row*2+col)
        # Block 0: (0,0), (0,1), (1,0), (1,1)
        # Block 1: (0,2), (0,3), (1,2), (1,3)
        # Block 2: (2,0), (2,1), (3,0), (3,1)
        # Block 3: (2,2), (2,3), (3,2), (3,3)
        
        br = (block // 2) * 2
        bc = (block % 2) * 2
        lr = local // 2
        lc = local % 2
        return (br + lr) * 4 + (bc + lc)
        
    def get_global_B_idx(block, local):
        # B is 4x4. Same structure.
        br = (block // 2) * 2
        bc = (block % 2) * 2
        lr = local // 2
        lc = local % 2
        return (br + lr) * 4 + (bc + lc)
        
    def get_global_C_idx(block, local):
        # C is 4x4. Same structure.
        br = (block // 2) * 2
        bc = (block % 2) * 2
        lr = local // 2
        lc = local % 2
        return (br + lr) * 4 + (bc + lc)

    col_offset = 0
    for t in range(7):
        for i in range(4):
            if U2[i][t] == 0: continue
            for j in range(4):
                if V2[j][t] == 0: continue
                # We have a product of A_block_i and B_block_j
                # Coefficients are U2[i][t] and V2[j][t]
                # This product is a 2x2 matrix, decomposed by 2x2 Strassen (k in 0..6)
                for k in range(7):
                    # Check if this 2x2 Strassen term is active
                    # We need to see if any U2, V2, W2 entries for this k are non-zero
                    # Since they are fixed, we can just iterate.
                    
                    # U part
                    u_coeff = U2[i][t]
                    for la in range(4):
                        if U2[la][k] == 0: continue
                        gu = get_global_A_idx(i, la)
                        U[gu][col_offset] += u_coeff * U2[la][k]
                        
                    # V part
                    v_coeff = V2[j][t]
                    for lb in range(4):
                        if V2[lb][k] == 0: continue
                        gv = get_global_B_idx(j, lb)
                        V[gv][col_offset] += v_coeff * V2[lb][k]
                        
                    # W part
                    # Which C blocks?
                    for c in range(4):
                        if W2[c][t] == 0: continue
                        w_coeff = W2[c][t]
                        for lc in range(4):
                            if W2[lc][k] == 0: continue
                            gw = get_global_C_idx(c, lc)
                            W[gw][col_offset] += w_coeff * W2[lc][k]
                            
                    col_offset += 1
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}

    # --- Part 2: Schoolbook 4x4x5 ---
    # B2 is 4x5. C2 is 4x5.
    # V rows for B2: 16..35 (4*4 + 0..19)
    # W rows for C2: 16..35 (4*4 + 0..19)
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                # U index: i*4 + j
                # V index: 4*4 + j*5 + k = 16 + j*5 + k
                # W index: 4*4 + k*4 + i = 16 + k*4 + i
                
                U[i*4 + j][col_offset] = 1
                V[16 + j*5 + k][col_offset] = 1
                W[16 + k*4 + i][col_offset] = 1
                col_offset += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
