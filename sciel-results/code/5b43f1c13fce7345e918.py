def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    # 1. Define 2x2x2 Strassen coefficients
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11 (Row-Major for A, B)
    # W Index: k*n+i -> 0:00, 1:10, 2:01, 3:11
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
    # W indices: 0(C00), 2(C01), 1(C10), 3(C11)
    W2[0, 0] = 1; W2[2, 0] = -1; W2[1, 0] = -1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += M2, C11 += M2
    W2[1, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 += M3
    W2[2, 2] = 1
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C10 += M4, C11 += M4
    W2[1, 3] = 1; W2[3, 3] = 1
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C01 += M5, C11 += M5
    W2[2, 4] = 1; W2[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C10 += M6
    W2[1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C01 += M7, C11 += M7
    W2[2, 6] = 1; W2[3, 6] = 1
    
    # 2. Construct 4x4x4 (Rank 49)
    # A is 4x4, viewed as 2x2 blocks of 2x2.
    # B is 4x4, viewed as 2x2 blocks of 2x2.
    # C is 4x4, viewed as 2x2 blocks of 2x2.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Outer Strassen coefficients (Block level)
    # Same structure as U2, V2, W2 but operating on blocks.
    # We iterate over the 7 outer terms (s).
    # For each s, we have coefficients for the 4 A-blocks and 4 B-blocks.
    # These are given by U2[:, s] and V2[:, s].
    # The "multiplication" is of the form (Sum U2[i,s] A_block_i) * (Sum V2[j,s] B_block_j).
    # This results in a sum of products A_block_i * B_block_j.
    # Each A_block_i * B_block_j is a 2x2x2 multiplication.
    # So we add the contribution of the 2x2x2 tensor (U2, V2, W2) scaled by U2[i,s]*V2[j,s].
    
    for s in range(7):
        # Coefficients for A blocks and B blocks from outer Strassen
        a_coeffs = U2[:, s] # 4x1
        b_coeffs = V2[:, s] # 4x1
        
        # For each pair of blocks (i, j)
        for i_block in range(4):
            if a_coeffs[i_block] == 0: continue
            for j_block in range(4):
                if b_coeffs[j_block] == 0: continue
                
                scale = a_coeffs[i_block] * b_coeffs[j_block]
                if scale == 0: continue
                
                # Determine global indices for A and B
                # A_block i_block corresponds to rows i_block//2, i_block//2+1 and cols i_block%2, i_block%2+1?
                # No, standard block indexing:
                # Block 0: rows 0-1, cols 0-1
                # Block 1: rows 0-1, cols 2-3
                # Block 2: rows 2-3, cols 0-1
                # Block 3: rows 2-3, cols 2-3
                
                r_start_a = (i_block // 2) * 2
                c_start_a = (i_block % 2) * 2
                r_start_b = (j_block // 2) * 2
                c_start_b = (j_block % 2) * 2
                
                # Add contribution of 2x2x2 Strassen to global U, V, W
                # The inner Strassen uses U2, V2, W2.
                # U2[k, t] is coeff of A_inner[k] in term t.
                # Global A index: (r_start_a + r_inner) * 4 + (c_start_a + c_inner)
                
                for t in range(7):
                    r_global = s * 7 + t
                    
                    # Update U4
                    for k in range(4):
                        if U2[k, t] == 0: continue
                        r_inner = k // 2
                        c_inner = k % 2
                        i_global = (r_start_a + r_inner) * 4 + (c_start_a + c_inner)
                        U4[i_global, r_global] += scale * U2[k, t]
                        
                    # Update V4
                    for k in range(4):
                        if V2[k, t] == 0: continue
                        r_inner = k // 2
                        c_inner = k % 2
                        i_global = (r_start_b + r_inner) * 4 + (c_start_b + c_inner)
                        V4[i_global, r_global] += scale * V2[k, t]
                        
                    # Update W4
                    # W2 index k corresponds to C_inner.
                    # C_inner indices: 0(0,0), 1(1,0), 2(0,1), 3(1,1) based on W2 definition above?
                    # My W2 definition:
                    # 0: C00, 1: C10, 2: C01, 3: C11.
                    # So k=0 -> r=0, c=0.
                    # k=1 -> r=1, c=0.
                    # k=2 -> r=0, c=1.
                    # k=3 -> r=1, c=1.
                    
                    for k in range(4):
                        if W2[k, t] == 0: continue
                        # Map W2 index k to r_inner, c_inner
                        if k == 0:
                            r_inner, c_inner = 0, 0
                        elif k == 1:
                            r_inner, c_inner = 1, 0
                        elif k == 2:
                            r_inner, c_inner = 0, 1
                        elif k == 3:
                            r_inner, c_inner = 1, 1
                        
                        i_global_r = r_start_a + r_inner # Wait, C block is at (i_block, j_block)?
                        # C = A * B.
                        # C_block (i_block, j_block) is the product of A_block (i_block) and B_block (j_block)?
                        # No.
                        # In the outer Strassen, the result C is a linear combination of the M_s.
                        # Each M_s is a block product.
                        # The block product A_block_i * B_block_j contributes to C_block (i, j)?
                        # No, standard matrix multiplication:
                        # C_{ij} = sum_k A_{ik} B_{kj}.
                        # In block form: C_{IJ} = sum_K A_{IK} B_{KJ}.
                        # The term A_block_i * B_block_j contributes to C_block (i, j) ONLY IF the block indices match the multiplication structure.
                        # Wait, the outer Strassen algorithm computes the 2x2 block matrix product.
                        # The "multiplication" M_s is a scalar (in the field of blocks).
                        # M_s = L_A(A_blocks) L_B(B_blocks).
                        # This M_s is a 2x2 matrix (block).
                        # It contributes to the final C_blocks via the W2 coefficients.
                        # W2[k, s] is the coefficient of M_s in C_block_k.
                        # So for a fixed s, the term M_s contributes to C_block_k with weight W2[k, s].
                        # M_s itself is a sum of products A_block_i * B_block_j.
                        # So the contribution of A_block_i * B_block_j to C_block_k is:
                        # scale * W2[k, s] * (Inner 2x2x2 tensor).
                        
                        # So, for a fixed s, i_block, j_block, t:
                        # We add to W4 at position corresponding to C_block_k (where k is the block index in 2x2 C).
                        # The C_block_k is located at rows k//2*2.., cols k%2*2..
                        
                        c_r_start = (k // 2) * 2
                        c_c_start = (k % 2) * 2
                        
                        i_global = (c_r_start + r_inner) * 4 + (c_c_start + c_inner)
                        # W index convention: W[k*n+i, r]. n=4.
                        # k is col, i is row.
                        # My W4 is 16x49.
                        # Index in W4: (col * 4 + row).
                        
                        W4[i_global, r_global] += scale * W2[k, t] * W2[k, s] # Wait.
                        # W2[k, s] is the outer coefficient.
                        # W2[k, t] is the inner coefficient.
                        # Yes.
                        
    # 3. Extend to 4x4x8 (Rank 98)
    U8 = np.zeros((16, 98), dtype=int)
    V8 = np.zeros((32, 98), dtype=int) # 4*8 = 32
    W8 = np.zeros((32, 98), dtype=int) # 4*8 = 32
    
    # First 49 terms: B cols 0-3, C cols 0-3
    U8[:, :49] = U4
    
    # V8 for first 49 terms:
    # V4 is 16x49.
    # V8 is 32x98.
    # Rows 0-15 of V8 correspond to B cols 0-3.
    # Rows 16-31 correspond to B cols 4-7.
    V8[:16, :49] = V4
    V8[16:, :49] = 0
    
    # W8 for first 49 terms:
    # W4 is 16x49.
    # W8 is 32x98.
    # Rows 0-15 of W8 correspond to C cols 0-3.
    W8[:16, :49] = W4
    W8[16:, :49] = 0
    
    # Next 49 terms: B cols 4-7, C cols 4-7
    U8[:, 49:] = U4
    
    # V8 for next 49 terms:
    V8[:16, 49:] = 0
    V8[16:, 49:] = V4
    
    # W8 for next 49 terms:
    W8[:16, 49:] = 0
    W8[16:, 49:] = W4
    
    return {
        "U": U8.tolist(),
        "V": V8.tolist(),
        "W": W8.tolist()
    }
