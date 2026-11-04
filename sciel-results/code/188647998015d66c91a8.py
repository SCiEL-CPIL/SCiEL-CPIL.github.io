import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # 1. Define 2x2x2 Strassen decomposition (rank 7)
    # U_22: (4, 7), V_22: (4, 7), W_22: (4, 7)
    U_22 = np.zeros((4, 7), dtype=int)
    V_22 = np.zeros((4, 7), dtype=int)
    W_22 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U_22[0, 0] += 1; U_22[3, 0] += 1
    V_22[0, 0] += 1; V_22[3, 0] += 1
    W_22[0, 0] += 1; W_22[3, 0] += 1
    
    # M2
    U_22[2, 1] += 1; U_22[3, 1] += 1
    V_22[0, 1] += 1
    W_22[2, 1] += 1
    
    # M3
    U_22[0, 2] += 1
    V_22[1, 2] += 1; V_22[3, 2] -= 1
    W_22[1, 2] += 1
    
    # M4
    U_22[3, 3] += 1
    V_22[2, 3] += 1; V_22[0, 3] -= 1
    W_22[0, 3] += 1; W_22[2, 3] += 1
    
    # M5
    U_22[0, 4] += 1; U_22[1, 4] += 1
    V_22[3, 4] += 1
    W_22[0, 4] -= 1; W_22[1, 4] += 1
    
    # M6
    U_22[2, 5] += 1; U_22[0, 5] -= 1
    V_22[0, 5] += 1; V_22[1, 5] += 1
    W_22[3, 5] += 1
    
    # M7
    U_22[1, 6] += 1; U_22[3, 6] -= 1
    V_22[2, 6] += 1; V_22[3, 6] += 1
    W_22[0, 6] += 1; W_22[3, 6] -= 1
    
    # 2. Build 4x4x4 decomposition (rank 49)
    # We will create a function to add a 4x4x4 Strassen block to global matrices
    def add_4x4x4_strassen(U, V, W, offset_A, offset_B, offset_C):
        # offset_A, offset_B, offset_C are starting indices for the 16 elements
        # A blocks: 00, 01, 10, 11 relative to offset_A
        # B blocks: 00, 01, 10, 11 relative to offset_B
        # C blocks: 00, 01, 10, 11 relative to offset_C
        
        # Strassen steps for 4x4 (composed of 2x2 blocks)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a 2x2x2 multiplication.
        # We iterate through the 7 steps.
        
        # Define block offsets
        # A00: 0, A01: 2, A10: 8, A11: 10 (within 4x4)
        # Wait, 4x4 indices:
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        # 12 13 14 15
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        A_offsets = [0, 2, 8, 10] # Start indices for A00, A01, A10, A11
        B_offsets = [0, 2, 8, 10]
        C_offsets = [0, 2, 8, 10]
        
        # Mapping of Strassen steps to block combinations
        # Step 0 (M1): (A00+A11) * (B00+B11) -> C00, C11
        # Step 1 (M2): (A10+A11) * B00 -> C10
        # Step 2 (M3): A00 * (B01-B11) -> C01
        # Step 3 (M4): A11 * (B10-B00) -> C00, C10
        # Step 4 (M5): (A00+A01) * B11 -> C00, C01
        # Step 5 (M6): (A10-A00) * (B00+B01) -> C11
        # Step 6 (M7): (A01-A11) * (B10+B11) -> C00, C11
        
        # We need to define for each step:
        # - Which A blocks and signs
        # - Which B blocks and signs
        # - Which C blocks and signs
        
        steps = [
            # (A_blocks, A_signs, B_blocks, B_signs, C_blocks, C_signs)
            ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # M1
            ([2, 3], [1, 1], [0], [1], [2], [1]),             # M2
            ([0], [1], [1, 3], [1, -1], [1], [1]),            # M3
            ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),      # M4
            ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),      # M5
            ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),      # M6
            ([1, 3], [1, -1], [2, 3], [1, 1], [0, 3], [1, -1]), # M7
        ]
        
        for s in range(7):
            A_blocks, A_signs, B_blocks, B_signs, C_blocks, C_signs = steps[s]
            
            # For this Strassen step, we use the 2x2x2 decomposition U_22[:, s]
            # But we must accumulate contributions from multiple blocks.
            
            # Actually, the 2x2x2 decomposition U_22[:, s] corresponds to the coefficients
            # for the 4 elements of the 2x2 matrix.
            # If the input is a sum of blocks, we add the corresponding U_22 columns
            # scaled by the signs.
            
            # However, U_22[:, s] is fixed for the 2x2x2 multiplication.
            # The "blocks" logic applies to how the global indices map to the local 2x2 indices.
            
            # Let's refine:
            # For a given Strassen step s, the "virtual" 2x2 multiplication is performed.
            # The inputs are linear combinations of the 2x2 blocks.
            # The U_22 matrix for step s (column s) gives the coefficients for the 4 local elements.
            
            # So, for global A:
            # For each local index l in 0..3:
            #   coeff = U_22[l, s]
            #   For each block b in A_blocks:
            #       global_idx = offset_A + A_offsets[b] + l
            #       U[global_idx, col] += coeff * A_signs[index_of_b]
            
            # Wait, A_signs corresponds to the blocks in A_blocks.
            # The order matters.
            
            for l in range(4):
                u_val = U_22[l, s]
                if u_val != 0:
                    for i_b, b in enumerate(A_blocks):
                        sign = A_signs[i_b]
                        global_idx = offset_A + A_offsets[b] + l
                        U[global_idx, col] += u_val * sign
                        
                v_val = V_22[l, s]
                if v_val != 0:
                    for i_b, b in enumerate(B_blocks):
                        sign = B_signs[i_b]
                        global_idx = offset_B + B_offsets[b] + l
                        V[global_idx, col] += v_val * sign
                        
                w_val = W_22[l, s]
                if w_val != 0:
                    for i_c, c in enumerate(C_blocks):
                        sign = C_signs[i_c]
                        global_idx = offset_C + C_offsets[c] + l
                        W[global_idx, col] += w_val * sign
                        
            col += 1
            
        return col

    # Initialize global matrices
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 * 2 + rank_441
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)
    W = np.zeros((36, total_rank), dtype=int)
    
    col = 0
    
    # First 4x4x4 part (B columns 0-3)
    # A is at offset 0
    # B is at offset 0 (first 4 columns of B)
    # C is at offset 0 (first 4 columns of C)
    col = add_4x4x4_strassen(U, V, W, 0, 0, 0)
    
    # Second 4x4x4 part (B columns 4-7)
    # A is at offset 0
    # B is at offset 4 (next 4 columns of B)
    # C is at offset 4 (next 4 columns of C)
    col = add_4x4x4_strassen(U, V, W, 0, 4, 4)
    
    # Third part: 4x4x1 (B column 8)
    # Schoolbook for 4x4x1
    # A (4x4) * b (4x1) -> c (4x1)
    # c_i = sum_j A_ij b_j
    # T[a, b, c] = 1 if a=i*4+j, b=j, c=i
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = 8 + j # B column 8, row j
            c_idx = 8 + i # C column 8, row i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    # Return lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
