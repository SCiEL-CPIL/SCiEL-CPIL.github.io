import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate factors for M_{2,2,2} (Strassen)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: i, j, k in 0..1
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A22-A11)(B11+B12)
    U2[3, 0] = 1; U2[2, 0] = -1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1 # C12 += M1, C22 -= M1? 
    # Let's use the standard Strassen formulas:
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 + M1 - M3 - M7
    
    # M1 = (A22 - A11)(B11 + B12)
    # A22 is A[1,1], A11 is A[0,0]
    # B11 is B[0,0], B12 is B[0,1]
    # U2[3, 0] = 1 (A22)
    # U2[0, 0] = -1 (A11)
    # V2[0, 0] = 1 (B11)
    # V2[1, 0] = 1 (B12)
    # W2[1, 0] = 1 (C12)
    # W2[3, 0] = -1 (C22)
    
    # M2 = (A11 + A12)B22
    # U2[0, 1] = 1, U2[1, 1] = 1
    # V2[3, 1] = 1
    # W2[1, 1] = 1 (C12)
    # W2[0, 1] = -1 (C11)
    
    # M3 = A11(B12 - B22)
    # U2[0, 2] = 1
    # V2[1, 2] = 1, V2[3, 2] = -1
    # W2[2, 2] = 1 (C21)
    # W2[3, 2] = -1 (C22)
    
    # M4 = A22(B11 + B22)
    # U2[3, 3] = 1
    # V2[0, 3] = 1, V2[3, 3] = 1
    # W2[0, 3] = 1 (C11)
    # W2[2, 3] = 1 (C21)
    
    # M5 = (A11 + A22)B11
    # U2[0, 4] = 1, U2[3, 4] = 1
    # V2[0, 4] = 1
    # W2[0, 4] = 1 (C11)
    # W2[3, 4] = 1 (C22)
    
    # M6 = (A21 - A11)(B11 + B21)
    # U2[2, 5] = 1, U2[0, 5] = -1
    # V2[0, 5] = 1, V2[2, 5] = 1
    # W2[0, 5] = 1 (C11)
    
    # M7 = (A12 + A22)(B21 - B22)
    # U2[1, 6] = 1, U2[3, 6] = 1
    # V2[2, 6] = 1, V2[3, 6] = -1
    # W2[3, 6] = -1 (C22)
    
    # Fill U2, V2, W2
    U2[3, 0] = 1; U2[0, 0] = -1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[1, 0] = 1; W2[3, 0] = -1
    
    U2[0, 1] = 1; U2[1, 1] = 1
    V2[3, 1] = 1
    W2[1, 1] = 1; W2[0, 1] = -1
    
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = -1
    
    U2[3, 3] = 1
    V2[0, 3] = 1; V2[3, 3] = 1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    U2[0, 4] = 1; U2[3, 4] = 1
    V2[0, 4] = 1
    W2[0, 4] = 1; W2[3, 4] = 1
    
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[0, 5] = 1
    
    U2[1, 6] = 1; U2[3, 6] = 1
    V2[2, 6] = 1; V2[3, 6] = -1
    W2[3, 6] = -1
    
    # 2. Compose to get M_{4,4,4} factors
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # The 7 multiplications are M1..M7.
    # Each Mk is computed using the 2x2 Strassen algorithm on the block entries.
    # This is equivalent to taking the tensor product of the Strassen factors?
    # No, it's a substitution.
    # We can generate U4, V4, W4 by iterating over the 7 "meta-multiplications"
    # and for each, iterating over the 7 "inner-multiplications".
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for meta_r in range(7):
        # For each meta multiplication, we have a specific U2, V2, W2 pattern
        # But the indices in U2, V2, W2 refer to block indices (0..3)
        # We need to map block indices to scalar indices (0..15)
        
        # The block indices for U (A) are:
        # 0: A00, 1: A01, 2: A10, 3: A11
        # Scalar index for block b: b * 4 + sub_idx
        # Wait, A is 4x4. Blocks are 2x2.
        # A[0,0] is A00. A[0,1] is A01. A[1,0] is A10. A[1,1] is A11.
        # In U2, index 0 is A00, 1 is A01, 2 is A10, 3 is A11.
        # In the 4x4 matrix, A00 is at rows 0,1 cols 0,1.
        # So scalar index for A00 sub-element (i,j) is i*4 + j.
        
        for inner_r in range(7):
            col = meta_r * 7 + inner_r
            
            # Get the coefficients from U2, V2, W2 for this meta/inner step
            # U2[local_u, meta_r] is the coefficient for A_block_local_u
            # V2[local_v, meta_r] is the coefficient for B_block_local_v
            # W2[local_w, meta_r] is the coefficient for C_block_local_w
            
            # For each non-zero in U2[:, meta_r], add to U4
            for local_u in range(4):
                coeff_u = U2[local_u, meta_r]
                if coeff_u != 0:
                    # local_u maps to block. 
                    # Block 0: A00 (rows 0,1, cols 0,1)
                    # Block 1: A01 (rows 0,1, cols 2,3)
                    # Block 2: A10 (rows 2,3, cols 0,1)
                    # Block 3: A11 (rows 2,3, cols 2,3)
                    
                    # The inner Strassen algorithm (inner_r) operates on the 2x2 block.
                    # So we need to know which sub-elements of the block are involved.
                    # This requires the U2, V2, W2 of the inner step too.
                    # Wait, the inner step is the SAME Strassen algorithm.
                    # So for a given block, we use the inner_r coefficients from U2, V2, W2.
                    
                    # So, for U4:
                    # The entry U4[scalar_idx, col] = coeff_u * U2_inner[local_sub_u, inner_r]
                    # where scalar_idx = block_offset + local_sub_u
                    
                    # We need the inner U2, V2, W2 for the inner_r step.
                    # But U2, V2, W2 are the same for all blocks.
                    
                    # So:
                    # For each local_u (block index in meta step):
                    #   For each local_sub_u (sub-index in inner step):
                    #       U4[block_offset(local_u) + local_sub_u, col] += coeff_u * U2[local_sub_u, inner_r]
                    
                    block_row = (local_u // 2) * 2
                    block_col = (local_u % 2) * 2
                    base_u = block_row * 4 + block_col
                    
                    for local_sub_u in range(4):
                        coeff_inner_u = U2[local_sub_u, inner_r]
                        if coeff_inner_u != 0:
                            sub_row = local_sub_u // 2
                            sub_col = local_sub_u % 2
                            scalar_idx = base_u + sub_row * 4 + sub_col
                            U4[scalar_idx, col] += coeff_u * coeff_inner_u

            # Similarly for V4
            for local_v in range(4):
                coeff_v = V2[local_v, meta_r]
                if coeff_v != 0:
                    block_row = (local_v // 2) * 2
                    block_col = (local_v % 2) * 2
                    base_v = block_row * 4 + block_col
                    
                    for local_sub_v in range(4):
                        coeff_inner_v = V2[local_sub_v, inner_r]
                        if coeff_inner_v != 0:
                            sub_row = local_sub_v // 2
                            sub_col = local_sub_v % 2
                            scalar_idx = base_v + sub_row * 4 + sub_col
                            V4[scalar_idx, col] += coeff_v * coeff_inner_v

            # Similarly for W4
            for local_w in range(4):
                coeff_w = W2[local_w, meta_r]
                if coeff_w != 0:
                    block_row = (local_w // 2) * 2
                    block_col = (local_w % 2) * 2
                    base_w = block_row * 4 + block_col
                    
                    for local_sub_w in range(4):
                        coeff_inner_w = W2[local_sub_w, inner_r]
                        if coeff_inner_w != 0:
                            sub_row = local_sub_w // 2
                            sub_col = local_sub_w % 2
                            scalar_idx = base_w + sub_row * 4 + sub_col
                            W4[scalar_idx, col] += coeff_w * coeff_inner_w

    # 3. Generate factors for M_{4,4,1} (Schoolbook)
    # Rank 16
    U5 = np.zeros((16, 16))
    V5 = np.zeros((4, 16)) # m=4, p=1
    W5 = np.zeros((4, 16)) # p=1, n=4
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U5[i * 4 + j, r] = 1
            V5[j, r] = 1
            W5[i, r] = 1 # W index k*n+i, k=0 here? 
            # Wait, M_{4,4,1} output is 4x1.
            # k=0. i=0..3.
            # W index is 0*4 + i = i.
            # So W5[i, r] = 1 is correct for the 4x1 matrix.
            
    # 4. Combine
    # U: (16, 65)
    # V: (20, 65)
    # W: (20, 65)
    
    U = np.zeros((16, 65))
    V = np.zeros((20, 65))
    W = np.zeros((20, 65))
    
    # Copy M_{4,4,4} parts
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # Copy M_{4,4,1} parts
    U[:, 49:] = U5
    # V5 is 4x16. Place in rows 16-19 of V (for k=4)
    V[16:, 49:] = V5
    # W5 is 4x16. Place in rows 16-19 of W (for k=4)
    W[16:, 49:] = W5
    
    # Convert to lists of lists of integers
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
