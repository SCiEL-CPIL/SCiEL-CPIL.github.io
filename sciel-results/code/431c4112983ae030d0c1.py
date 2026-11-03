import time
import sys

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    
    # Helper to add a term to the decomposition lists
    # U, V, W are lists of dicts or lists. 
    # We will build the final matrices U, V, W as lists of lists.
    # But first, let's collect the terms as tuples of (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # Actually, it's easier to build the matrices directly column by column.
    
    rank_4x4_strassen = 49
    rank_rest = 16 + 16 + 4 + 4 + 1 + 1
    total_rank = rank_4x4_strassen + rank_rest
    # 91
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Top-Left 4x4 Block using Strassen Recursion ---
    # We need to define the 2x2 Strassen terms first.
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    
    # Term definitions for 2x2: (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # u_indices: which A elements are involved
    # v_indices: which B elements are involved
    # w_indices: which C elements are affected (0:00, 1:01, 2:10, 3:11)
    
    strassen_2x2_terms = [
        # P1: (A00+A11)(B00+B11) -> C00+1, C11+1
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        # P2: (A10+A11)B00 -> C10+1
        ((2, 3), (1, 1), (0,), (1,), (2,), (1,)),
        # P3: A00(B01-B11) -> C01+1
        ((0,), (1,), (1, 3), (1, -1), (1,), (1,)),
        # P4: A11(B10-B00) -> C00+1, C10+1
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        # P5: (A00+A01)B11 -> C00-1, C01+1
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        # P6: (A10-A00)(B00+B01) -> C11+1
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # P7: (A01-A11)(B10+B11) -> C00+1, C11-1
        ((1, 3), (1, -1), (2, 3), (1, 1), (0, 3), (1, -1)),
    ]
    
    # Now, use these to build 4x4 terms.
    # 4x4 Strassen:
    # Blocks: A00, A01, A10, A11 (each 2x2)
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 (4x4) = P1 + P4 - P5 + P7
    # C01 (4x4) = P3 + P5
    # C10 (4x4) = P2 + P4
    # C11 (4x4) = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2 matrix product. We expand each Pk into 7 scalar products.
    # We need to map the "block indices" of the 4x4 matrices to the "scalar indices" of the 2x2 matrices.
    # Block 00 of 4x4 contains scalar indices 0,1,2,3 of the 2x2 sub-block?
    # Let's define the mapping.
    # 4x4 Matrix A:
    # 0  1  4  5
    # 2  3  6  7
    # 8  9 12 13
    #10 11 14 15
    #
    # Block A00 (top-left 2x2): indices 0,1,2,3.
    # Block A01 (top-right 2x2): indices 4,5,6,7.
    # Block A10 (bottom-left 2x2): indices 8,9,10,11.
    # Block A11 (bottom-right 2x2): indices 12,13,14,15.
    
    # For a given block b (0..3), the local indices 0..3 map to global offsets.
    def block_offsets(b):
        if b == 0: return [0, 1, 2, 3]
        if b == 1: return [4, 5, 6, 7]
        if b == 2: return [8, 9, 10, 11]
        if b == 3: return [12, 13, 14, 15]
        
    # For each of the 7 super-terms Pk in 4x4:
    # We determine the linear combination of 2x2 blocks for U and V.
    # Then, for each scalar term in the 2x2 expansion of that product, we add to the global U, V, W.
    
    # Super Term 1: (A00+A11)(B00+B11)
    # U blocks: 0 and 3 with coeff 1. V blocks: 0 and 3 with coeff 1.
    # Super Term 2: (A10+A11)B00
    # U blocks: 2 and 3 with coeff 1. V blocks: 0 with coeff 1.
    # Super Term 3: A00(B01-B11)
    # U blocks: 0 with coeff 1. V blocks: 1 with coeff 1, 3 with coeff -1.
    # Super Term 4: A11(B10-B00)
    # U blocks: 3 with coeff 1. V blocks: 2 with coeff 1, 0 with coeff -1.
    # Super Term 5: (A00+A01)B11
    # U blocks: 0 and 1 with coeff 1. V blocks: 3 with coeff 1.
    # Super Term 6: (A10-A00)(B00+B01)
    # U blocks: 2 with coeff 1, 0 with coeff -1. V blocks: 0 and 1 with coeff 1.
    # Super Term 7: (A01-A11)(B10+B11)
    # U blocks: 1 with coeff 1, 3 with coeff -1. V blocks: 2 and 3 with coeff 1.
    
    super_terms_4x4 = [
        # (u_block_coeffs, v_block_coeffs)
        # u_block_coeffs: list of (block_id, coeff)
        (((0, 1), (3, 1)), ((0, 1), (3, 1))), # P1
        (((2, 1), (3, 1)), ((0, 1),)),       # P2
        (((0, 1),), ((1, 1), (3, -1))),      # P3
        (((3, 1),), ((2, 1), (0, -1))),      # P4
        (((0, 1), (1, 1)), ((3, 1),)),       # P5
        (((2, 1), (0, -1)), ((0, 1), (1, 1))), # P6
        (((1, 1), (3, -1)), ((2, 1), (3, 1))), # P7
    ]
    
    # Output mapping for 4x4 Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Note: C00, C01, C10, C11 are 2x2 blocks of the 4x4 result C.
    # We need to map these back to scalar contributions.
    
    # Let's iterate through each super term Pk (k=0..6)
    for k in range(7):
        u_blocks, v_blocks = super_terms_4x4[k]
        
        # For each scalar term in the 2x2 Strassen decomposition:
        for term_idx in range(7):
            u_loc_idx, u_loc_coeff, v_loc_idx, v_loc_coeff, w_loc_idx, w_loc_coeff = strassen_2x2_terms[term_idx]
            
            # Expand u_loc to global U indices
            # u_loc_idx are local indices 0..3 in a 2x2 block.
            # But wait, the super term U is a sum of 2x2 blocks.
            # Each block has 4 elements.
            # So for each block in u_blocks, we add the local pattern scaled by the block coeff.
            
            # This approach is getting complex to trace.
            # Simpler: Just generate the 49 terms by iterating all 4x4 "scalar" positions?
            # No, Strassen is recursive.
            
            # Let's do it explicitly.
            # U_global is a 16-dim vector.
            # V_global is a 16-dim vector.
            # W_global is a 16-dim vector (for the 4x4 block).
            
            # We will construct the 49 terms one by one.
            
            pass # Placeholder
    
    # Given the complexity and risk of bug in manual recursion mapping, 
    # and the 150s limit, I will fall back to a **Greedy Random Search** 
    # to try to reduce the rank from 125 towards 91 or lower.
    # If I can't find a valid reduction, I return the schoolbook (125).
    # However, 91 is guaranteed. I should try to implement the 91 correctly.
    
    # Alternative: Hardcode the 91 decomposition if I can trust my logic?
    # Risky.
    
    # Let's try a simpler structural improvement: 
    # Use Strassen for 2x2 blocks in the 5x5 matrix directly?
    # 5 = 2+2+1.
    # Blocks: 2x2, 2x2, 1x1.
    # Too many mixed sizes.
    
    # I will implement the 91-rank decomposition carefully.
    
    def add_term_4x4(u_block_coeffs, v_block_coeffs, w_block_coeffs_4x4, current_col):
        # u_block_coeffs: list of (block_id, coeff) for U (4x4)
        # v_block_coeffs: list of (block_id, coeff) for V (4x4)
        # w_block_coeffs_4x4: dict mapping 2x2 block id in C (0:TL, 1:TR, 2:BL, 3:BR) to coeff
        # This function adds ONE scalar term from the expansion of the block products.
        
        # First, we must expand the block products into scalar products.
        # This function is actually adding a TERM of the 4x4 decomposition.
        # But a "term" in 4x4 Strassen is a 2x2 matrix product.
        # We need to expand that 2x2 product into 7 scalar products.
        
        # So this function should be called for each of the 7 scalar expansions of a super-term.
        # Inputs:
        # u_scalar_coeffs: list of (global_A_index, coeff)
        # v_scalar_coeffs: list of (global_B_index, coeff)
        # w_scalar_coeffs: list of (global_W_index, coeff)
        
        for idx, coeff in u_scalar_coeffs:
            U[idx][current_col] += coeff
        for idx, coeff in v_scalar_coeffs:
            V[idx][current_col] += coeff
        for idx, coeff in w_scalar_coeffs:
            W[idx][current_col] += coeff
            
    # We need to generate the 49 terms.
    # Let's do it loop over k (0..6) and t (0..6).
    for k in range(7):
        u_blocks, v_blocks = super_terms_4x4[k]
        for t in range(7):
            u_loc_idx, u_loc_coeff, v_loc_idx, v_loc_coeff, w_loc_idx, w_loc_coeff = strassen_2x2_terms[t]
            
            # Construct global U indices and coeffs
            u_global = []
            for b_id, b_coeff in u_blocks:
                offsets = block_offsets(b_id)
                for local_i, coeff_i in zip(u_loc_idx, u_loc_coeff):
                    # local_i is 0..3 in the 2x2 block
                    # global index is offsets[local_i]
                    u_global.append((offsets[local_i], b_coeff * coeff_i))
            
            v_global = []
            for b_id, b_coeff in v_blocks:
                offsets = block_offsets(b_id)
                for local_j, coeff_j in zip(v_loc_idx, v_loc_coeff):
                    v_global.append((offsets[local_j], b_coeff * coeff_j))
            
            # W mapping:
            # The 2x2 result block from this scalar term contributes to specific 2x2 blocks of the 4x4 C.
            # w_loc_idx are 0..3 in the 2x2 result.
            # We need to know which 2x2 block of the 4x4 C this super-term Pk contributes to, and with what sign.
            
            # Super Term Pk contribution to 4x4 C blocks:
            if k == 0: # P1
                # C00 +1, C11 +1
                w_map = {0: 1, 3: 1}
            elif k == 1: # P2
                # C10 +1
                w_map = {2: 1}
            elif k == 2: # P3
                # C01 +1
                w_map = {1: 1}
            elif k == 3: # P4
                # C00 +1, C10 +1
                w_map = {0: 1, 2: 1}
            elif k == 4: # P5
                # C00 -1, C01 +1
                w_map = {0: -1, 1: 1}
            elif k == 5: # P6
                # C11 +1
                w_map = {3: 1}
            elif k == 6: # P7
                # C00 +1, C11 -1
                w_map = {0: 1, 3: -1}
            
            w_global = []
            for local_w, coeff_w in zip(w_loc_idx, w_loc_coeff):
                # local_w is 0..3 in the 2x2 result block.
                # Which 2x2 block of 4x4 C is this?
                # The super term Pk is a 2x2 matrix.
                # Its elements are added to the 4x4 C blocks specified in w_map.
                # Wait, Pk is the RESULT of the block multiplication.
                # So the elements of Pk (0..3) map to the same relative positions in the target 4x4 C blocks.
                
                # If w_map says Pk goes to Block 0 (TL) with coeff 1:
                # Then local element 0 (TL of Pk) goes to global TL of Block 0.
                # local element 1 (TR of Pk) goes to global TR of Block 0.
                # etc.
                
                for target_block, target_coeff in w_map.items():
                    offsets = block_offsets(target_block)
                    global_idx = offsets[local_w]
                    w_global.append((global_idx, target_coeff * coeff_w))
            
            # Add the term
            for idx, coeff in u_global:
                U[idx][col] += coeff
            for idx, coeff in v_global:
                V[idx][col] += coeff
            for idx, coeff in w_global:
                W[idx][col] += coeff
                
            col += 1
            
    # --- Part 2: Remaining 42 terms (Schoolbook) ---
    # Indices 4 (row/col 4) are involved.
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # A44 B44 is already covered? No, A44 B44 is part of the top-left 4x4 block?
    # Wait. My 4x4 Strassen covered A[0:4, 0:4] * B[0:4, 0:4].
    # This produces C[0:4, 0:4] = A[0:4, 0:4]B[0:4, 0:4] + A[0:4, 4:5]B[4:5, 0:4]?
    # NO. The 4x4 Strassen decomposition computes the product of two 4x4 matrices.
    # It does NOT include the cross terms from the 5x5 expansion.
    # The 5x5 product is:
    # C[0:4, 0:4] = A[0:4, 0:4]B[0:4, 0:4] + A[0:4, 4]B[4, 0:4]
    # My 49 terms compute ONLY A[0:4, 0:4]B[0:4, 0:4].
    # So I still need to add the cross terms.
    
    # Cross terms:
    # 1. A[0:4, 4]B[4, 0:4] -> C[0:4, 0:4]
    #    A_col_4 (4x1) * B_row_4 (1x4). Rank 16.
    # 2. A[0:4, 4]B[4, 4] -> C[0:4, 4]
    #    A_col_4 (4x1) * B[4,4] (1x1). Rank 4.
    # 3. A[4, 0:4]B[0:4, 0:4] -> C[4, 0:4]
    #    A_row_4 (1x4) * B[0:4, 0:4] (4x4). Rank 4.
    # 4. A[4, 0:4]B[0:4, 4] -> C[4, 4]
    #    A_row_4 (1x4) * B_col_4 (4x1). Rank 1.
    # 5. A[4, 4]B[4, 0:4] -> C[4, 0:4]
    #    A[4,4] (1x1) * B_row_4 (1x4). Rank 4.
    # 6. A[4, 4]B[4, 4] -> C[4, 4]
    #    Rank 1.
    
    # Total cross terms: 16 + 4 + 4 + 1 + 4 + 1 = 30.
    # Plus the 49 = 79?
    # Wait, earlier I calculated 91.
    # 49 (TL) + 16 (A41 B14) + 16 (A44 B41) + 4 (A41 B11) + 4 (A14 B44) + 1 (A14 B41) + 1 (A11 B11).
    # The terms I listed now:
    # 1. A41 B14 (16) - Matches "1"
    # 2. A41 B11 (4) - Matches "2" (A_col_4 * B[4,4] is A41 B11? No. A[0:4, 4] is A41. B[4, 4] is B11. Yes.)
    # 3. A14 B44 (4) - Matches "3" (A_row_4 is A14. B[0:4, 0:4] is B44. Yes.)
    # 4. A14 B41 (1) - Matches "4" (A_row_4 is A14. B_col_4 is B41. Yes.)
    # 5. A11 B44 (4) - Matches "5" (A[4,4] is A11. B_row_4 is B44? No, B[4, 0:4] is B14? 
    #    Let's check indices.
    #    B is 5x5.
    #    B[0:4, 0:4] is B44.
    #    B[0:4, 4] is B41.
    #    B[4, 0:4] is B14.
    #    B[4, 4] is B11.
    
    # So:
    # A11 (1x1) * B14 (1x4) -> C[4, 0:4]. Rank 4.
    # A11 (1x1) * B11 (1x1) -> C[4, 4]. Rank 1.
    
    # So the list is:
    # A41 B14 (16)
    # A41 B11 (4)
    # A14 B44 (4)
    # A14 B41 (1)
    # A11 B14 (4)
    # A11 B11 (1)
    
    # Sum: 16+4+4+1+4+1 = 30.
    # Total Rank = 49 + 30 = 79.
    
    # This is even better!
    
    # Let's add these 30 terms.
    
    # Helper to add schoolbook terms for specific indices
    def add_schoolbook_terms(a_indices, b_indices, c_indices, start_col):
        nonlocal col
        for i in a_indices:
            for j in b_indices:
                # A_i * B_j -> C_k
                # Need to map i, j to k.
                # This is only valid if it's a simple product.
                pass
    
    # I will just loop manually.
    
    # 1. A41 B14 (16 terms)
    # A indices: 0*5+4, 1*5+4, 2*5+4, 3*5+4 (Col 4 of A) -> 4, 9, 14, 19
    # B indices: 4*5+0, 4*5+1, 4*5+2, 4*5+3 (Row 4 of B) -> 20, 21, 22, 23
    # C indices: i*5+j for i in 0..3, j in 0..3.
    for i in range(4):
        for j in range(4):
            a_idx = i * 5 + 4
            b_idx = 4 * 5 + j
            c_idx = j * 5 + i # W index is k*n+i, where k is col of B (j) and i is row of A.
            # Wait. C = A B.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # Here j=4 is the summation index? No.
            # A41 is A[0:4, 4]. B14 is B[4, 0:4].
            # Product is A[0:4, 4] * B[4, 0:4].
            # Result is C[0:4, 0:4].
            # C[i, k] gets contribution A[i, 4] * B[4, k].
            # i in 0..3, k in 0..3.
            # W index for C[i, k] is k*5 + i.
            for i in range(4):
                for k in range(4):
                    a_idx = i * 5 + 4
                    b_idx = 4 * 5 + k
                    w_idx = k * 5 + i
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # 2. A41 B11 (4 terms)
    # A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        a_idx = i * 5 + 4
        b_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i # C[i, 4] -> k=4, i=i.
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # 3. A14 B44 (4 terms)
    # A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    for k in range(4):
        a_idx = 4 * 5 + k
        b_idx = k * 5 + 0 # B[k, 0]? No.
        # A[4, j] * B[j, k] -> C[4, k].
        # j is the summation index.
        # We are doing schoolbook, so each term is one j.
        for j in range(4):
            a_idx = 4 * 5 + j
            b_idx = j * 5 + k
            w_idx = k * 5 + 4
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # 4. A14 B41 (1 term? No, 4 terms)
    # A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    for j in range(4):
        a_idx = 4 * 5 + j
        b_idx = j * 5 + 4
        w_idx = 4 * 5 + 4
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # 5. A11 B14 (4 terms)
    # A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    for k in range(4):
        a_idx = 4 * 5 + 4
        b_idx = 4 * 5 + k
        w_idx = k * 5 + 4
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # 6. A11 B11 (1 term)
    a_idx = 4 * 5 + 4
    b_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    U[a_idx][col] = 1
    V[b_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Total col should be 79.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
