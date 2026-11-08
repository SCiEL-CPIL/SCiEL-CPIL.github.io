import numpy as np

DIMENSIONS = [4, 4, 6]

def get_strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2, C11 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += M3, C11 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 += M4, C10 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5, C01 += M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += M7
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We construct U, V, W as lists of lists
    # Rank 49 for 4x4x4 part, Rank 28 for 4x4x2 part
    # Total rank 77
    
    U = []
    V = []
    W = []
    
    U2, V2, W2 = get_strassen_222()
    
    # Part 1: 4x4x4 (k=0..3)
    # Recursive Strassen
    # We need to map the 2x2 block indices to 4x4 indices
    # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # Wait, A is 4x4. Row-major.
    # A00: rows 0,1 cols 0,1 -> 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> 10,11,14,15
    
    # B blocks: B00 (0,1,8,9), B01 (2,3,10,11), B10 (4,5,12,13), B11 (6,7,14,15)
    # B is 4x4.
    # B00: rows 0,1 cols 0,1 -> 0,1,8,9
    # B01: rows 0,1 cols 2,3 -> 2,3,10,11
    # B10: rows 2,3 cols 0,1 -> 4,5,12,13
    # B11: rows 2,3 cols 2,3 -> 6,7,14,15
    
    # C blocks: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
    # C is 4x4.
    # C00: rows 0,1 cols 0,1 -> 0,1,4,5
    # C01: rows 0,1 cols 2,3 -> 2,3,6,7
    # C10: rows 2,3 cols 0,1 -> 8,9,12,13
    # C11: rows 2,3 cols 2,3 -> 10,11,14,15
    
    # For each of the 7 Strassen steps in the 4x4x4 block:
    # We have linear forms on A_blocks and B_blocks.
    # Each linear form is a sum of 2x2 matrices.
    # We decompose each 2x2 matrix product using the 2x2x2 Strassen decomposition.
    
    # Let's define the 7 Strassen combinations for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Mk, we have a product of two 2x2 matrices.
    # Let LA be the linear form on A blocks (2x2 matrix of coefficients)
    # Let LB be the linear form on B blocks (2x2 matrix of coefficients)
    # The product is LA * LB.
    # This is a 2x2x2 tensor scaled by the block positions.
    
    # We can just use the 2x2x2 Strassen decomposition for each Mk.
    # The 2x2x2 decomposition U2, V2, W2 works on 4 entries.
    # We need to map the 4 entries of the block to the 16 entries of the 4x4 matrix.
    
    # Helper to expand a 2x2 block decomposition to 4x4
    def expand_block(U_block, V_block, W_block, a_offset, b_offset, c_offset):
        # U_block is (4, 7)
        # a_offset is the starting index of the 2x2 block in the 4x4 matrix
        # But we have 4 blocks in A.
        # The linear form LA is a sum of blocks.
        # This is getting complicated to do generically.
        
        # Alternative: Just hardcode the 49 terms?
        # Or use numpy to build the full tensors and decompose?
        # No, we need the sparse structure.
        
        pass

    # Let's try a different approach.
    # Build the full 4x4x4 Strassen decomposition explicitly.
    
    # The 7 block products are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B01-B11)
    # 4. A11(B10-B00)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    
    # For each product, we use the 2x2x2 Strassen decomposition.
    # The 2x2x2 decomposition has 7 terms.
    # Total 49 terms.
    
    # We define the block indices
    # A blocks: 0:00, 1:01, 2:10, 3:11
    # B blocks: 0:00, 1:01, 2:10, 3:11
    # C blocks: 0:00, 1:01, 2:10, 3:11
    
    # For each block product, we determine the coefficients for A and B blocks.
    # Then we apply the 2x2x2 Strassen decomposition.
    
    # 2x2x2 Strassen U2, V2, W2
    # U2[i, k] is coeff of A_i in term k
    # V2[j, k] is coeff of B_j in term k
    # W2[l, k] is coeff of C_l in term k
    
    # For a block product with A-coeffs a_coeffs (4x1) and B-coeffs b_coeffs (4x1):
    # The term k of the 2x2x2 decomposition contributes:
    # U_new = sum_i a_coeffs[i] * U2[i, k] * (block expansion)
    # This is not quite right.
    
    # Let's just generate the 49 terms by iterating.
    
    # Block offsets in 4x4 matrix (row-major)
    # Block 0 (00): 0, 1, 4, 5
    # Block 1 (01): 2, 3, 6, 7
    # Block 2 (10): 8, 9, 12, 13
    # Block 3 (11): 10, 11, 14, 15
    
    block_offsets = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # Define the 7 Strassen combinations for 4x4x4
    # Each is a tuple (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict block -> coeff
    # b_coeffs: dict block -> coeff
    # c_coeffs: dict block -> coeff (for the sum of C blocks)
    
    combos = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}), # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}), # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}), # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}), # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}), # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})  # M7
    ]
    
    # For each combo, we have a 2x2x2 problem.
    # We use the 2x2x2 Strassen decomposition (U2, V2, W2).
    # For each term k in 0..6:
    # The A-part is sum over blocks i where a_coeffs[i] != 0:
    #   a_coeffs[i] * U2[local_idx, k] * (expansion of block i)
    
    # We will build U, V, W lists.
    
    for a_c, b_c, c_c in combos:
        for k in range(7):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16
            
            # A part
            for block_idx, coeff in a_c.items():
                # local index in 2x2 block
                # U2 is 4x7. U2[local, k]
                # local 0,1,2,3 correspond to block entries
                # We need to map local index to global index
                # But U2[local, k] is the coefficient for the local entry.
                # The block entry is expanded to 4 global entries.
                # Wait, U2[local, k] is a scalar.
                # The local entry corresponds to one of the 4 positions in the 2x2 block.
                # We need to add coeff * U2[local, k] to the corresponding global indices.
                
                # Which local index?
                # U2 is defined for the 2x2 matrix.
                # The 2x2 matrix has entries 0,1,2,3.
                # These correspond to the 4 entries of the block.
                # So for local l in 0..3:
                #   val = coeff * U2[l, k]
                #   if val != 0:
                #       global_idx = block_offsets[block_idx][l]
                #       u_vec[global_idx] += val
                for l in range(4):
                    val = coeff * U2[l, k]
                    if val != 0:
                        global_idx = block_offsets[block_idx][l]
                        u_vec[global_idx] += val
                        
            # B part
            for block_idx, coeff in b_c.items():
                for l in range(4):
                    val = coeff * V2[l, k]
                    if val != 0:
                        global_idx = block_offsets[block_idx][l]
                        v_vec[global_idx] += val
                        
            # C part
            # W2[l, k] is the coefficient for C_local in term k.
            # The C block contribution is sum over blocks c where c_c[c] != 0:
            #   c_c[c] * W2[local_c, k]
            # Wait, W2 is for the 2x2 C matrix.
            # The C block is a 2x2 matrix.
            # So for local l in 0..3:
            #   val = W2[l, k]
            #   if val != 0:
            #       for each C block c_idx with coeff c_coeff:
            #           global_idx = block_offsets[c_idx][l]
            #           w_vec[global_idx] += c_coeff * val
            for l in range(4):
                val = W2[l, k]
                if val != 0:
                    for c_idx, c_coeff in c_c.items():
                        global_idx = block_offsets[c_idx][l]
                        w_vec[global_idx] += c_coeff * val
                        
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    # Part 2: 4x4x2 (k=4,5)
    # B is 4x2.
    # We treat this as 4 blocks of 2x2x2?
    # No, B is 4x2.
    # Partition A into 2x2 blocks.
    # Partition B into two 2x2 blocks? No, B is 4x2.
    # B_top (2x2), B_bot (2x2).
    # C_top (2x2), C_bot (2x2).
    # C_top = A00 B_top + A01 B_bot
    # C_bot = A10 B_top + A11 B_bot
    
    # This is 4 products of 2x2x2.
    # 1. A00 B_top -> C_top
    # 2. A01 B_bot -> C_top
    # 3. A10 B_top -> C_bot
    # 4. A11 B_bot -> C_bot
    
    # For each product, we use 2x2x2 Strassen (7 terms).
    # Total 28 terms.
    
    # B indices:
    # B_top: rows 0,1 cols 0,1 -> 0,1,8,9 (in 4x2? No, B is 4x2)
    # B is 4x2. Row-major.
    # Row 0: 0,1
    # Row 1: 2,3
    # Row 2: 4,5
    # Row 3: 6,7
    # B_top (rows 0,1): 0,1,2,3
    # B_bot (rows 2,3): 4,5,6,7
    
    # C indices:
    # C is 4x2.
    # C_top (rows 0,1): 0,1,2,3
    # C_bot (rows 2,3): 4,5,6,7
    
    # A blocks same as before.
    
    b_offsets = {
        'top': [0, 1, 2, 3],
        'bot': [4, 5, 6, 7]
    }
    
    c_offsets = {
        'top': [0, 1, 2, 3],
        'bot': [4, 5, 6, 7]
    }
    
    # Products:
    # (A00, B_top, C_top)
    # (A01, B_bot, C_top)
    # (A10, B_top, C_bot)
    # (A11, B_bot, C_bot)
    
    products_442 = [
        (0, 'top', 'top'), # A00, B_top, C_top
        (1, 'bot', 'top'), # A01, B_bot, C_top
        (2, 'top', 'bot'), # A10, B_top, C_bot
        (3, 'bot', 'bot')  # A11, B_bot, C_bot
    ]
    
    for a_block, b_part, c_part in products_442:
        for k in range(7):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 8 # B is 4x2 = 8 entries
            w_vec = [0.0] * 8 # C is 4x2 = 8 entries
            
            # A part
            for l in range(4):
                val = U2[l, k]
                if val != 0:
                    global_idx = block_offsets[a_block][l]
                    u_vec[global_idx] += val
                    
            # B part
            for l in range(4):
                val = V2[l, k]
                if val != 0:
                    global_idx = b_offsets[b_part][l]
                    v_vec[global_idx] += val
                    
            # C part
            for l in range(4):
                val = W2[l, k]
                if val != 0:
                    global_idx = c_offsets[c_part][l]
                    w_vec[global_idx] += val
                    
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    # Return U, V, W
    # U shape (16, 77)
    # V shape (24, 77) -- Wait, V is (m*p, rank) = (4*6, 77) = (24, 77)
    # My V vectors for Part 1 are length 16 (for 4x4 B).
    # My V vectors for Part 2 are length 8 (for 4x2 B).
    # I need to pad them to length 24.
    
    # Part 1 V vectors: indices 0..15 correspond to B cols 0..3.
    # Part 2 V vectors: indices 0..7 correspond to B cols 4,5?
    # No, B is 4x6.
    # B cols 0,1,2,3 are in Part 1.
    # B cols 4,5 are in Part 2.
    # In Part 2, I used B as 4x2.
    # The indices 0..7 in my Part 2 V vectors correspond to the 8 entries of the 4x2 matrix.
    # These should map to global indices 16..23?
    # B row-major:
    # Row 0: 0,1,2,3,4,5
    # Row 1: 6,7,8,9,10,11
    # Row 2: 12,13,14,15,16,17
    # Row 3: 18,19,20,21,22,23
    
    # Part 1 B is 4x4. Indices 0..15.
    # Part 2 B is 4x2. Indices 16..23.
    
    # I need to shift the Part 2 V vectors by 16.
    
    # Let's rebuild V properly.
    
    V_final = []
    for i in range(len(V)):
        if i < 49:
            # Part 1
            v = V[i] + [0.0]*8
        else:
            # Part 2
            v = [0.0]*16 + V[i]
        V_final.append(v)
        
    # W is (p*n, rank) = (6*4, 77) = (24, 77)
    # Part 1 W is 4x4. Indices 0..15.
    # Part 2 W is 4x2. Indices 16..23.
    # C row-major:
    # Row 0: 0,1,2,3,4,5
    # ...
    
    W_final = []
    for i in range(len(W)):
        if i < 49:
            w = W[i] + [0.0]*8
        else:
            w = [0.0]*16 + W[i]
        W_final.append(w)
        
    return {"U": U, "V": V_final, "W": W_final}

