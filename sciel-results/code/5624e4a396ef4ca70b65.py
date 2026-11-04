import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 + 49 = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: index in U (0..15)
    # v_idx: index in V (0..31)
    # w_idx: index in W (0..31)
    # coeff: coefficient
    # col: current column index in the factor matrices
    
    # We will construct the decomposition for C = A * B
    # B is 4x8. We split B into B1 (4x4) and B2 (4x4).
    # C is 4x8. We split C into C1 (4x4) and C2 (4x4).
    # C1 = A * B1
    # C2 = A * B2
    
    # Strassen for 4x4 * 4x4
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # The 7 products M1..M7 for 4x4 Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each Mk is a 2x2 matrix product.
    # We decompose each Mk using 2x2 Strassen (7 terms).
    
    # 2x2 Strassen terms for X * Y = Z:
    # S1 = (X00 + X11) * (Y00 + Y11) -> Z00 += S1, Z11 += S1
    # S2 = (X10 + X11) * Y00       -> Z10 += S2, Z11 -= S2
    # S3 = X00 * (Y01 - Y11)       -> Z01 += S3, Z11 += S3
    # S4 = X11 * (Y10 - Y00)       -> Z00 += S4, Z10 += S4
    # S5 = (X00 + X01) * Y11       -> Z00 -= S5, Z01 += S5
    # S6 = (X10 - X00) * (Y00 + Y01) -> Z11 += S6
    # S7 = (X01 - X11) * (Y10 + Y11) -> Z00 += S7
    
    # We need to map the 2x2 block elements to global indices.
    # For a 4x4 matrix M, block (bi, bj) contains elements M[2*bi + u, 2*bj + v] for u,v in 0,1.
    # Global index for A (4x4): row * 4 + col.
    # Global index for B (4x8): row * 8 + col.
    # Global index for C (4x8): row * 4 + col? 
    # Wait, W index is k*n + i. n=4.
    # W index for C[row, col] is col * 4 + row.
    
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff

    # Define the 7 sub-terms for a 2x2 Strassen product
    # Returns list of (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # u_coeff_dict: { (bi, bj, ui, vi): coeff } for the 2x2 block X
    # v_coeff_dict: { (bi, bj, vi, wi): coeff } for the 2x2 block Y
    # w_coeff_dict: { (ci, cj, wi, wj): coeff } for the 2x2 block Z
    
    # Actually, it's easier to just generate the 7 linear combinations for X and Y,
    # and the 4 linear combinations for Z, and link them.
    
    # Let's pre-compute the 7 Strassen steps for 2x2.
    # Each step has:
    # L: linear combo of X elements
    # R: linear combo of Y elements
    # W_contrib: dict of (z_row, z_col) -> coeff
    
    strassen_2x2_steps = [
        # (L_coeffs, R_coeffs, W_coeffs)
        # L_coeffs: list of (x_idx, coeff) where x_idx is 0..3 for 2x2 block (00, 01, 10, 11)
        # R_coeffs: list of (y_idx, coeff)
        # W_coeffs: list of (z_idx, coeff)
        
        # S1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # S2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # S3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # S4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # S5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # S6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # S7
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    # Now we need to map 2x2 block indices to global indices.
    # For a 4x4 matrix, block (bi, bj) of size 2x2.
    # Element (ui, vi) in block (bi, bj) is at global row 2*bi + ui, col 2*bj + vi.
    
    # We will iterate over the two halves of B.
    for half in range(2):
        # half 0: B1, C1
        # half 1: B2, C2
        b_col_offset = half * 4
        c_col_offset = half * 4
        
        # Iterate over the 7 Strassen steps for 4x4
        # We need to define the 7 combinations of 4x4 blocks.
        # Let's define them as lists of (block_idx, coeff)
        # Block idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        strassen_4x4_steps = [
            # (L_blocks, R_blocks)
            # L_blocks: list of (block_idx, coeff)
            # R_blocks: list of (block_idx, coeff)
            
            # M1 = (A00 + A11) * (B00 + B11)
            ( [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
            # M2 = (A10 + A11) * B00
            ( [(2, 1), (3, 1)], [(0, 1)] ),
            # M3 = A00 * (B01 - B11)
            ( [(0, 1)], [(1, 1), (3, -1)] ),
            # M4 = A11 * (B10 - B00)
            ( [(3, 1)], [(2, 1), (0, -1)] ),
            # M5 = (A00 + A01) * B11
            ( [(0, 1), (1, 1)], [(3, 1)] ),
            # M6 = (A10 - A00) * (B00 + B01)
            ( [(2, 1), (0, -1)], [(0, 1), (1, 1)] ),
            # M7 = (A01 - A11) * (B10 + B11)
            ( [(1, 1), (3, -1)], [(2, 1), (3, 1)] ),
        ]
        
        # We need to know which C blocks are updated by each M_k.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will store the contributions to C blocks for each M_k.
        # c_contribs[k] = list of (c_block_idx, coeff)
        c_contribs = [
            [(0, 1), (3, 1)], # M1
            [(2, 1), (3, -1)], # M2
            [(1, 1), (3, 1)], # M3
            [(0, 1), (2, 1)], # M4
            [(0, -1), (1, 1)], # M5
            [(3, 1)], # M6
            [(0, 1)], # M7
        ]
        
        col_idx = half * 49
        
        for k in range(7):
            L_blocks = strassen_4x4_steps[k][0]
            R_blocks = strassen_4x4_steps[k][1]
            
            # We form two linear combinations of 2x2 blocks: L_comb and R_comb.
            # L_comb is a 2x2 matrix.
            # R_comb is a 2x2 matrix.
            # We compute L_comb * R_comb using 2x2 Strassen.
            
            # We need to represent L_comb and R_comb as collections of elements.
            # Each element in L_comb is a linear combination of A elements.
            # Each element in R_comb is a linear combination of B elements.
            
            # Let's build the 4 elements of L_comb (2x2).
            # L_comb[ui, vi] = sum(coeff * A[2*bi+ui, 2*bj+vi]) for (bi,bj,coeff) in L_blocks
            # Note: L_blocks contains block indices.
            # Block 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            L_elements = [[0 for _ in range(4)] for _ in range(4)] # 4x4? No, 2x2 block elements.
            # Actually, we just need to add to U.
            # U index for A[row, col] is row * 4 + col.
            
            # Let's just iterate over the 7 sub-steps of 2x2 Strassen for this M_k.
            for sub_k in range(7):
                L_sub_coeffs = strassen_2x2_steps[sub_k][0]
                R_sub_coeffs = strassen_2x2_steps[sub_k][1]
                W_sub_coeffs = strassen_2x2_steps[sub_k][2]
                
                # L_sub_coeffs is for the 2x2 block L_comb.
                # It specifies linear combos of the 4 elements of L_comb.
                # But L_comb elements themselves are linear combos of A blocks.
                # This is getting nested.
                
                # Alternative:
                # For this specific M_k and sub_k, we can determine the global U, V, W indices directly.
                
                # U part:
                # The input to the 2x2 Strassen step is L_comb and R_comb.
                # L_comb is formed from A blocks.
                # The 2x2 Strassen step uses specific elements of L_comb.
                # e.g., L_sub_coeffs[0] is (0, 1) meaning L_comb[0,0].
                # L_comb[0,0] is sum of A[2*bi, 2*bj] for (bi,bj,coeff) in L_blocks.
                
                # So, for each element (ui, vi) of L_comb used in this sub-step:
                # For each (bi, bj, coeff_block) in L_blocks:
                #   For each (ui_sub, vi_sub) in the 2x2 block (bi, bj):
                #      Global A index: (2*bi + ui_sub) * 4 + (2*bj + vi_sub)
                #      Coeff: coeff_block * coeff_sub
                #      Add to U[global_A_idx][col_idx]
                
                # Similarly for V.
                # R_comb is formed from B blocks.
                # B blocks are in the specific half.
                # Global B index: (2*bi + ui_sub) * 8 + (2*bj + vi_sub + b_col_offset)
                
                # Similarly for W.
                # W part:
                # The output is Z_comb, which contributes to C blocks.
                # The 2x2 Strassen step produces Z_comb elements.
                # W_sub_coeffs specifies which elements of Z_comb are used.
                # e.g., W_sub_coeffs[0] is (0, 1) meaning Z_comb[0,0].
                # Z_comb[0,0] contributes to C blocks specified in c_contribs[k].
                # For each (c_block_idx, coeff_c) in c_contribs[k]:
                #   c_block_idx corresponds to C block (ci, cj).
                #   The element Z_comb[0,0] corresponds to C[2*ci + 0, 2*cj + 0].
                #   Wait, the mapping of Z_comb elements to C blocks is 1-to-1 for the block structure.
                #   Z_comb is a 2x2 matrix.
                #   C_block is a 2x2 matrix.
                #   So Z_comb[ui, vi] adds to C[2*ci + ui, 2*cj + vi].
                
                # Let's implement this logic.
                
                # 1. Process U
                for ui_sub, coeff_sub in L_sub_coeffs:
                    # ui_sub is 0..3? No, strassen_2x2_steps uses 0..3 for 2x2 elements.
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    # We need to map ui_sub to (bi, bj) in L_blocks?
                    # No, L_sub_coeffs refers to the elements of the *combined* 2x2 matrix L_comb.
                    # L_comb[0,0] is formed from A blocks.
                    # We need to know which A blocks contribute to L_comb[ui_sub, vi_sub].
                    
                    # Let's restructure.
                    # Pre-compute for each of the 7 M_k, and for each of the 4 elements of L_comb and R_comb,
                    # the list of (global_idx, coeff) for U and V.
                    
                    pass # We'll do this in the loop below.
                    
                # This nested logic is complex. Let's flatten it.
                
                # For a given k (4x4 step) and sub_k (2x2 step):
                # We want to add 1 column to U, V, W.
                
                # U contributions:
                # L_sub_coeffs tells us which elements of L_comb are used.
                # Let's find the global U indices.
                for ui_sub_idx, coeff_sub in L_sub_coeffs:
                    # ui_sub_idx is 0..3.
                    # Map to (u_block_row, u_block_col) in 2x2 L_comb?
                    # No, L_comb is 2x2. ui_sub_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                    u_row_in_comb = ui_sub_idx // 2
                    u_col_in_comb = ui_sub_idx % 2
                    
                    # Which A blocks contribute to L_comb[u_row_in_comb, u_col_in_comb]?
                    # L_blocks is list of (block_idx, coeff_block).
                    # block_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                    for block_idx, coeff_block in L_blocks:
                        bi = block_idx // 2
                        bj = block_idx % 2
                        
                        # The element L_comb[u_row_in_comb, u_col_in_comb] is the sum of
                        # A[2*bi + u_row_in_comb, 2*bj + u_col_in_comb] * coeff_block
                        # for all blocks in L_blocks?
                        # No, L_comb is a 2x2 matrix.
                        # L_comb[i, j] = sum_{blocks} coeff_block * A_block[i, j]
                        # So for a fixed (i, j) in L_comb, the contribution from block (bi, bj) is
                        # coeff_block * A[2*bi + i, 2*bj + j].
                        
                        # So for this specific (u_row_in_comb, u_col_in_comb):
                        a_row = 2 * bi + u_row_in_comb
                        a_col = 2 * bj + u_col_in_comb
                        u_idx = a_row * 4 + a_col
                        total_coeff = coeff_block * coeff_sub
                        U[u_idx][col_idx] += total_coeff

                # V contributions
                for vi_sub_idx, coeff_sub in R_sub_coeffs:
                    v_row_in_comb = vi_sub_idx // 2
                    v_col_in_comb = vi_sub_idx % 2
                    
                    for block_idx, coeff_block in R_blocks:
                        bi = block_idx // 2
                        bj = block_idx % 2
                        
                        b_row = 2 * bi + v_row_in_comb
                        b_col = 2 * bj + v_col_in_comb + b_col_offset
                        v_idx = b_row * 8 + b_col
                        total_coeff = coeff_block * coeff_sub
                        V[v_idx][col_idx] += total_coeff

                # W contributions
                for wi_sub_idx, coeff_sub in W_sub_coeffs:
                    w_row_in_comb = wi_sub_idx // 2
                    w_col_in_comb = wi_sub_idx % 2
                    
                    # This element of Z_comb contributes to C blocks in c_contribs[k].
                    for c_block_idx, coeff_c in c_contribs[k]:
                        ci = c_block_idx // 2
                        cj = c_block_idx % 2
                        
                        c_row = 2 * ci + w_row_in_comb
                        c_col = 2 * cj + w_col_in_comb + c_col_offset
                        # W index: col * n + row
                        w_idx = c_col * 4 + c_row
                        total_coeff = coeff_c * coeff_sub
                        W[w_idx][col_idx] += total_coeff

                col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
