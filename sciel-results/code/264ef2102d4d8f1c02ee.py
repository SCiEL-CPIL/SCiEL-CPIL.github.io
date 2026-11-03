import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # We compute C = A @ B where A is 4x4, B is 4x8.
    # Split B into B_L (4x4) and B_R (4x4).
    # C_L = A @ B_L, C_R = A @ B_R.
    # Both are 4x4x4 multiplications.
    # Use Strassen for 4x4x4. Rank is 49 per product.
    # Total rank = 98.
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define 2x2 Strassen factors
    # U2, V2, W2 shape (4, 7)
    # Indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11
    U2[0, 0] = 1; U2[3, 0] = 1
    # V: B00, B11
    V2[0, 0] = 1; V2[3, 0] = 1
    # W: C00, C11 (from C00=M1+..., C11=M1-...)
    # Wait, W2 should map the sub-result to the C blocks.
    # In the 2x2 case, the result of the 7 multipliers is combined to form C.
    # But here we are decomposing the tensor.
    # The tensor T for 2x2x2 has T[i,j,k] = 1.
    # The decomposition is C[i,k] = sum_r W[i,k,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    # Let's check the W2 values for 2x2 Strassen.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # r=0 (M1): C00 += 1, C11 += 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00 is idx 0 (i=0,k=0), C11 is idx 3 (i=1,k=1) in 2x2?
    # Indices for W in 2x2: (k, i).
    # C[i,k] -> W[k, i, r].
    # C[0,0] -> W[0,0]. C[1,1] -> W[1,1].
    # My previous U2/V2/W2 indexing in the seed was a bit messy. Let's be precise.
    # U index: i*m + j. For 2x2, m=2.
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # V index: j*p + k. For 2x2, p=2.
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # W index: k*n + i. For 2x2, n=2.
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # Note: W index is k*n+i.
    # C[0,0] -> k=0, i=0 -> idx 0.
    # C[0,1] -> k=1, i=0 -> idx 2? No, k*n+i = 1*2+0 = 2.
    # C[1,0] -> k=0, i=1 -> idx 1.
    # C[1,1] -> k=1, i=1 -> idx 3.
    
    # Let's re-fill W2 based on C[i,k] -> W[k, i] (flattened k*n+i).
    # M1: C00, C11
    W2[0, 0] = 1 # C00 (k=0, i=0)
    W2[3, 0] = 1 # C11 (k=1, i=1)
    
    # M2: C10
    # C[1,0] -> k=0, i=1 -> idx 1
    W2[1, 1] = 1
    
    # M3: C01
    # C[0,1] -> k=1, i=0 -> idx 2
    W2[2, 2] = 1
    
    # M4: C10, C00
    # C[1,0] -> idx 1, C[0,0] -> idx 0
    W2[1, 3] = 1
    W2[0, 3] = 1
    
    # M5: C01, C00
    # C[0,1] -> idx 2, C[0,0] -> idx 0 (with -1)
    W2[2, 4] = 1
    W2[0, 4] = -1
    
    # M6: C11
    # C[1,1] -> idx 3
    W2[3, 5] = 1
    
    # M7: C00, C11
    # C[0,0] -> idx 1 (with +1), C[1,1] -> idx 3 (with -1)
    W2[0, 6] = 1
    W2[3, 6] = -1
    
    # Now we have U2, V2, W2 for 2x2x2.
    # We need to build 4x4x4 Strassen.
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11.
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11.
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11.
    
    # The 4x4 Strassen uses 7 multipliers M_k (each 2x2).
    # M_k = (Linear Combo of A blocks) * (Linear Combo of B blocks)
    # Each M_k is a 2x2 matrix product, so we decompose it using U2, V2, W2.
    
    # Mapping of block indices to 4x4 indices.
    # A block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # A index i*m+j.
    # B block (bj, bk) covers rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
    # B index j*p+k.
    # C block (bi, bk) covers rows 2*bi..2*bi+1, cols 2*bk..2*bk+1.
    # C index k*n+i.
    
    # Define the 7 block-level multipliers for 4x4.
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each M_k, we have:
    # A_part = sum_a coef_a * A_block_a
    # B_part = sum_b coef_b * B_block_b
    # M_k = A_part * B_part
    
    # We decompose M_k using the 2x2 Strassen.
    # The "A" in the 2x2 Strassen is the 2x2 matrix A_part.
    # The "B" in the 2x2 Strassen is the 2x2 matrix B_part.
    # The "C" in the 2x2 Strassen is the 2x2 matrix M_k.
    
    # We need to map the 2x2 Strassen indices to the 4x4 indices.
    # For a specific M_k:
    # U4 entries:
    # For each sub-r (0..6) in U2:
    #   U2[i_sub, j_sub, r_sub] is non-zero.
    #   This means A_sub[i_sub, j_sub] is used.
    #   A_sub is A_part.
    #   A_part = sum_a coef_a * A_block_a.
    #   So A_sub[i_sub, j_sub] = sum_a coef_a * A_block_a[i_sub, j_sub].
    #   A_block_a[i_sub, j_sub] corresponds to global A index (2*bi_a + i_sub, 2*bj_a + j_sub).
    #   So U4[global_i, global_j, r] += U2[i_sub, j_sub, r_sub] * coef_a.
    
    # Similarly for V4 and W4.
    
    # Let's define the block coefficients for each M_k.
    # M_k: A_blocks with coeffs, B_blocks with coeffs.
    # A_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_blocks = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    col = 0
    for m_idx in range(7):
        a_coeffs, b_coeffs = strassen_blocks[m_idx]
        # a_coeffs: dict block_idx -> coeff
        # b_coeffs: dict block_idx -> coeff
        
        # For each sub-term in 2x2 Strassen (r_sub 0..6)
        for r_sub in range(7):
            r = col + r_sub
            
            # Update U
            # U2[i_sub, j_sub, r_sub]
            for i_sub in range(2):
                for j_sub in range(2):
                    u_val = U2[i_sub, j_sub, r_sub]
                    if u_val != 0:
                        # This sub-entry A_part[i_sub, j_sub] is used.
                        # A_part = sum_a coef_a * A_block_a
                        # So we add u_val * coef_a to U4[global_i, global_j, r]
                        # global_i = 2*bi_a + i_sub
                        # global_j = 2*bj_a + j_sub
                        for a_idx, coef in a_coeffs.items():
                            bi_a = a_idx // 2
                            bj_a = a_idx % 2
                            global_i = 2 * bi_a + i_sub
                            global_j = 2 * bj_a + j_sub
                            U[global_i * m + global_j, r] += u_val * coef
            
            # Update V
            for j_sub in range(2):
                for k_sub in range(2):
                    v_val = V2[j_sub, k_sub, r_sub]
                    if v_val != 0:
                        for b_idx, coef in b_coeffs.items():
                            bj_b = b_idx // 2
                            bk_b = b_idx % 2
                            global_j = 2 * bj_b + j_sub
                            global_k = 2 * bk_b + k_sub
                            V[global_j * p + global_k, r] += v_val * coef
            
            # Update W
            # W2[k_sub, i_sub, r_sub] maps to C_sub[i_sub, k_sub]
            # C_sub is the result of A_part * B_part.
            # In the 4x4 context, C_sub is part of C_block_c.
            # The 4x4 Strassen combines M_k to form C.
            # C = sum_k W_block[k, r_sub] * M_k
            # We need the block-level W coefficients.
            
            # Let's get the block-level W coefficients for M_k.
            # From 2x2 Strassen C formulas:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6 + M7
            
            # We need a matrix W_block where W_block[c_block_idx, m_idx] = coeff.
            # c_block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # m_idx: 0..6
            
            w_block = np.zeros((4, 7))
            # M1 (0)
            w_block[0, 0] = 1 # C00
            w_block[3, 0] = 1 # C11
            # M2 (1)
            w_block[2, 1] = 1 # C10
            # M3 (2)
            w_block[1, 2] = 1 # C01
            # M4 (3)
            w_block[2, 3] = 1 # C10
            w_block[0, 3] = 1 # C00
            # M5 (4)
            w_block[1, 4] = 1 # C01
            w_block[0, 4] = -1 # C00
            # M6 (5)
            w_block[3, 5] = 1 # C11
            # M7 (6)
            w_block[0, 6] = 1 # C00
            w_block[3, 6] = -1 # C11
            
            # For the current m_idx, we have W2[k_sub, i_sub, r_sub].
            # This contributes to C_sub[i_sub, k_sub].
            # C_sub is part of C_block_c.
            # C_block_c = sum_m W_block[c, m] * M_m
            # So C_sub[i_sub, k_sub] = sum_m W_block[c, m] * M_m[i_sub, k_sub]
            # We are decomposing M_m[i_sub, k_sub] using W2.
            # So the total contribution to C[global_i, global_k] is:
            # sum_c W_block[c, m_idx] * W2[k_sub, i_sub, r_sub] * (1)
            # Wait, W2 already includes the signs for the sub-terms?
            # Yes, W2[k_sub, i_sub, r_sub] is the coefficient for the sub-result.
            # So for a fixed m_idx, the contribution to C_block_c is:
            # W_block[c, m_idx] * (sum_{k_sub, i_sub} W2[k_sub, i_sub, r_sub] * M_m_sub[i_sub, k_sub])
            # This is correct.
            
            for k_sub in range(2):
                for i_sub in range(2):
                    w_sub_val = W2[k_sub, i_sub, r_sub]
                    if w_sub_val != 0:
                        for c_idx, coef in [(0, w_block[0, m_idx]), (1, w_block[1, m_idx]), (2, w_block[2, m_idx]), (3, w_block[3, m_idx])]:
                            if coef != 0:
                                bi_c = c_idx // 2
                                bk_c = c_idx % 2
                                global_k = 2 * bk_c + k_sub
                                global_i = 2 * bi_c + i_sub
                                W[global_k * n + global_i, r] += w_sub_val * coef
                                
            col += 1
            
    # Now we have the factors for C_L = A @ B_L.
    # We need C_R = A @ B_R.
    # This is identical in structure, but B_R corresponds to columns 4..7 of B.
    # B index j*p+k. For B_R, k is 4..7? No, B is 4x8.
    # B_L is cols 0..3, B_R is cols 4..7.
    # In the 4x4x4 decomposition for C_L, V indices were j*4+k (k in 0..3).
    # For C_R, V indices should be j*4 + (k+4).
    # We can just copy the V factors and shift the indices?
    # No, V is (m*p, rank) = (32, 98).
    # For C_L, we filled V[0..15, 0..48]. (j in 0..3, k in 0..3 -> idx j*8+k? No, V shape is m*p = 4*8=32).
    # Wait, V index is j*p+k. p=8.
    # So for B_L (k=0..3), indices are j*8 + k.
    # For B_R (k=4..7), indices are j*8 + (k+4).
    # My previous V filling used `global_j * p + global_k`.
    # In the loop, global_k was 0..3. So indices were j*8 + 0..3.
    # This is correct for B_L.
    # For C_R, we need to fill V[j*8 + 4..7, 49..97].
    # And W for C_R?
    # C_R is columns 4..7 of C.
    # C index k*n+i. For C_R, k is 4..7.
    # So W indices are (4..7)*4 + i.
    # The U factors are the same for C_L and C_R?
    # Yes, A is the same.
    # So U[0..15, 49..97] = U[0..15, 0..48].
    # V[0..15, 49..97] = V[0..15, 0..48] but with k shifted?
    # No, V depends on B.
    # V for C_R: B_R.
    # The structure is the same, just different B columns.
    # So V[j*8 + 4 + k_sub, r] = V[j*8 + k_sub, r-49]?
    # Let's just replicate the construction for the second half.
    
    # Copy U
    U[:, 49:] = U[:, 0:49]
    
    # For V, we need to map B_R indices.
    # In the first pass, we used global_k in 0..3.
    # For the second pass, we use global_k in 4..7.
    # So V[j*8 + 4 + k_sub, 49 + r_sub] = V[j*8 + k_sub, r_sub]?
    # Let's verify.
    # V index is j*p+k.
    # First pass: k in 0..3.
    # Second pass: k in 4..7.
    # The coefficients are the same.
    # So we can just copy V[:, 0:49] to V[:, 49:] but shift the row indices?
    # No, the row indices are different.
    # V[:, 0:49] has non-zeros in rows j*8 + k (k in 0..3).
    # V[:, 49:] should have non-zeros in rows j*8 + k (k in 4..7).
    # So we can construct V_R by taking V_L and shifting rows by 4?
    # Row index = j*8 + k.
    # Shift k by 4 means row index becomes j*8 + k + 4.
    # So V[:, 49:] = V[:, 0:49] with rows shifted by 4.
    # We can do this by creating a new array or just copying and adjusting.
    # Easier: just re-run the loop for the second half.
    
    # But we can optimize.
    # V[:, 49:] = np.zeros_like(V[:, 0:49])
    # For each row r in V[:, 0:49], if it's non-zero, copy to r+4 in V[:, 49:].
    # Since the pattern is regular, we can just do:
    # V[:, 49:] = V[:, 0:49]
    # Then for each column, shift the values? No, the rows are different.
    # Let's just re-run the filling logic for the second half.
    
    # Actually, simpler:
    # V[:, 49:] = V[:, 0:49]
    # Then, for each row i in 0..31:
    #   if i < 16 (k in 0..3):
    #      V[i+4, 49:] = V[i, 0:49] ? No.
    #      V[i, 0:49] corresponds to k in 0..3.
    #      We want V[i+4, 49:] to correspond to k in 4..7.
    #      So V[i+4, 49:] = V[i, 0:49].
    #   if i >= 16:
    #      V[i, 49:] = V[i, 0:49] (if i corresponds to k in 0..3 of second half? No)
    #      Wait, j*8+k.
    #      j=0, k=0..3 -> 0..3
    #      j=0, k=4..7 -> 4..7
    #      j=1, k=0..3 -> 8..11
    #      j=1, k=4..7 -> 12..15
    #      So for each j, rows 8j..8j+3 map to 8j+4..8j+7.
    #      So V[8j+4..8j+7, 49:] = V[8j..8j+3, 0:49].
    
    for j in range(4):
        V[8*j+4 : 8*j+8, 49:] = V[8*j : 8*j+4, 0:49]
        
    # For W, C_R is columns 4..7.
    # C index k*n+i.
    # First pass: k in 0..3.
    # Second pass: k in 4..7.
    # W[4..7, 49:] = W[0..3, 0:49]?
    # W index is k*n+i.
    # k=0..3 -> 0..15.
    # k=4..7 -> 16..31.
    # So W[16..31, 49:] = W[0..15, 0:49].
    
    W[16:32, 49:] = W[0:16, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
