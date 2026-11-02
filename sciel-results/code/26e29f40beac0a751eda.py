import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # p = 7, so we split B into B1 (4x4) and B2 (4x3)
    p1 = 4
    p2 = 3
    
    rank1 = 49 # Strassen for 4x4x4
    rank2 = 48 # Schoolbook for 4x4x3
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: A (4x4) * B1 (4x4) ---
    # Strassen's algorithm
    # Blocks of A: A00, A01, A10, A11 (2x2 each)
    # Blocks of B1: B00, B01, B10, B11 (2x2 each)
    
    # Strassen steps for 2x2x2
    # Returns list of (coeff_A, coeff_B, coeff_C, local_i, local_j, local_k) ?
    # Actually, we need to generate the rank-1 terms for the 4x4x4 product.
    
    # The 7 macro steps:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_s is a 2x2 matrix product.
    # We use Strassen (rank 7) to compute each M_s.
    
    # Strassen 2x2x2 decomposition details:
    # To compute X*Y where X,Y are 2x2:
    # S1 = X00+X11, S2 = X10+X11, S3 = X00, S4 = X11, S5 = X00+X01, S6 = X10-X00, S7 = X01-X11
    # T1 = Y00+Y11, T2 = Y00, T3 = Y01-Y11, T4 = Y10-Y00, T5 = Y11, T6 = Y00+Y01, T7 = Y10+Y11
    # P1 = S1*T1
    # P2 = S2*T2
    # P3 = S3*T3
    # P4 = S4*T4
    # P5 = S5*T5
    # P6 = S6*T6
    # P7 = S7*T7
    #
    # Result C:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to the global U, V, W.
    
    # Helper to add a rank-1 term
    # term: (u_idx, v_idx, w_idx, coeff)
    # u_idx is index in U (n*m)
    # v_idx is index in V (m*p)
    # w_idx is index in W (p*n)
    
    col = 0
    
    # Define the 7 macro steps as (list of A-block coeffs, list of B-block coeffs)
    # A-blocks: 00, 01, 10, 11
    # B-blocks: 00, 01, 10, 11
    
    macro_steps = [
        # (A_coeffs, B_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Result assembly coefficients for C blocks (00, 01, 10, 11) from M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    result_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},              # C01
        {1: 1, 3: 1},              # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # Strassen 2x2x2 sub-steps
    # X_coeffs, Y_coeffs for the 7 products P1..P7
    sub_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # P1
        ({2: 1, 3: 1}, {0: 1}),       # P2
        ({0: 1}, {1: 1, 3: -1}),      # P3
        ({3: 1}, {2: 1, 0: -1}),      # P4
        ({0: 1, 1: 1}, {3: 1}),       # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # P7
    ]
    
    # Result assembly for 2x2 product C from P1..P7
    sub_result_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},              # C01
        {1: 1, 3: 1},              # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # Iterate over macro steps
    for s in range(7):
        A_macro_coeffs = macro_steps[s][0]
        B_macro_coeffs = macro_steps[s][1]
        
        # Iterate over sub steps
        for t in range(7):
            A_sub_coeffs = sub_steps[t][0]
            B_sub_coeffs = sub_steps[t][1]
            
            # Combine coefficients for A blocks
            # A_block index: 0(00), 1(01), 2(10), 3(11)
            # Each block is 2x2.
            # We need to form the linear combination of A elements.
            
            # For this rank-1 term, we have a specific A linear comb and B linear comb.
            # The "product" is of two 2x2 matrices.
            # This product is further decomposed into rank-1 terms?
            # NO. The sub_strassen already decomposes the 2x2x2 product into 7 rank-1 terms.
            # So for each macro step s, we have 7 sub-terms.
            # Each sub-term corresponds to a specific rank-1 component of the 2x2x2 multiplication.
            
            # Wait, the logic is:
            # C = sum_s coeff_cs * (X_s Y_s)
            # X_s Y_s = sum_t coeff_ct * (U_t X_s V_t Y_s W_t) ??
            # No.
            # The tensor for 2x2x2 is:
            # T_{ij, kl, mn} = sum_t U_{ij,t} V_{kl,t} W_{mn,t}
            # So X Y = sum_t (U_t X) (V_t Y) W_t?
            # Actually, the rank-1 term is:
            # Term_t = (sum_{i,j} U_{ij,t} X_{ij}) * (sum_{k,l} V_{kl,t} Y_{kl}) * W_{mn,t}
            # This is not quite right.
            # The standard tensor decomposition is:
            # C_{mn} = sum_t (sum_{i,k} U_{ik,t} X_{ik}) (sum_{k,l} V_{kl,t} Y_{kl}) ... no.
            
            # Let's use the explicit formula.
            # For 2x2x2, the 7 terms are defined by the linear forms.
            # Term t contributes to C_{mn} with coefficient W_{mn,t}.
            # The input to the term is (L_t(X), M_t(Y)).
            # The term value is L_t(X) * M_t(Y) * W_{mn,t}.
            
            # So, for macro step s:
            # X_s = sum_{b} A_sb * A_b
            # Y_s = sum_{b} B_sb * B_b
            # X_s Y_s = sum_{t} L_t(X_s) M_t(Y_s) W_{mn,t}
            # L_t(X_s) = sum_{i,j} U_{ij,t} X_{s,ij}
            # X_{s,ij} = sum_{b} A_{sb} A_{b,ij}
            # So L_t(X_s) = sum_{b} A_{sb} (sum_{i,j} U_{ij,t} A_{b,ij})
            # This is a linear combination of A elements.
            
            # We can just compute the coefficients for each A element and B element.
            
            # U matrix index: i*m + j. i,j in 0..3.
            # V matrix index: j*p + k. j in 0..3, k in 0..3 (for B1).
            # W matrix index: k*n + i. k in 0..3, i in 0..3.
            
            # For this specific rank-1 term (s, t):
            # It contributes to C blocks.
            # The coefficient for C_{block_c, local_mn} is:
            # coeff_macro[s, block_c] * W_sub[t, local_mn]
            
            # The linear form for A is:
            # sum_{b} A_macro[s, b] * (sum_{i,j} U_sub[t, i, j] A_{b, i, j})
            # = sum_{b, i, j} A_macro[s, b] * U_sub[t, i, j] * A_{b, i, j}
            
            # We need to extract U_sub, V_sub, W_sub for 2x2x2.
            
            # Let's hardcode the U, V, W for 2x2x2 Strassen.
            # This is tedious but doable.
            
            # Alternatively, we can just build the 49 terms directly by iterating.
            
            # Let's define the U, V, W for 2x2x2 Strassen.
            # U, V, W are 4x7 matrices.
            
            # This is getting complex to do manually in the head.
            # I will use a known library or implementation if possible, but I must write it from scratch.
            
            # Let's try a simpler approach:
            # Use the schoolbook for 4x4x4? Rank 64.
            # Total rank 64 + 48 = 112. Same as seed.
            
            # I need to implement Strassen correctly.
            
            # Let's define the 2x2x2 Strassen tensors.
            # Reference:
            # U = [[1,1,0,0,0,0,0],
            #      [0,0,1,0,1,0,0],
            #      [0,0,0,1,0,1,0],
            #      [1,1,0,0,0,0,0]] ? No.
            
            # Let's use the formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            
            # We can represent this as a sum of 7 terms.
            # Each term t has:
            # A-linear form L_t(A)
            # B-linear form M_t(B)
            # C-coefficients W_t(C)
            
            # Term 1:
            # L_1 = A00+A11
            # M_1 = B00+B11
            # W_1: C00+=1, C11+=1
            
            # Term 2:
            # L_2 = A10+A11
            # M_2 = B00
            # W_2: C10+=1, C11-=1
            
            # Term 3:
            # L_3 = A00
            # M_3 = B01-B11
            # W_3: C01+=1, C11+=1
            
            # Term 4:
            # L_4 = A11
            # M_4 = B10-B00
            # W_4: C00+=1, C10+=1
            
            # Term 5:
            # L_5 = A00+A01
            # M_5 = B11
            # W_5: C00-=1, C01+=1
            
            # Term 6:
            # L_6 = A10-A00
            # M_6 = B00+B01
            # W_6: C11+=1
            
            # Term 7:
            # L_7 = A01-A11
            # M_7 = B10+B11
            # W_7: C00+=1
            
            # This is the 2x2x2 decomposition.
            
            # Now, for 4x4x4:
            # We have 7 macro terms.
            # Each macro term s uses a linear combo of A-blocks and B-blocks.
            # Let A_blocks be A00, A01, A10, A11.
            # Let B_blocks be B00, B01, B10, B11.
            
            # For macro term s:
            # X_s = sum_b cA[s,b] A_b
            # Y_s = sum_b cB[s,b] B_b
            
            # X_s Y_s is computed using the 2x2x2 decomposition above.
            # So X_s Y_s = sum_t L_t(X_s) M_t(Y_s) W_t
            
            # L_t(X_s) = sum_{i,j} U_2x2[t, i, j] X_{s, i, j}
            # X_{s, i, j} = sum_b cA[s,b] A_{b, i, j}
            # So L_t(X_s) = sum_b cA[s,b] (sum_{i,j} U_2x2[t, i, j] A_{b, i, j})
            
            # This is a linear combination of the 16 elements of A.
            # Similarly for B.
            
            # The contribution to C is:
            # sum_t W_2x2[t, c, m, n] * L_t(X_s) * M_t(Y_s)
            # where c is the block index in C (00, 01, 10, 11).
            
            # So for each s (0..6) and t (0..6), we have a rank-1 term for the 4x4x4 product.
            # Total 49 terms.
            
            # We need to construct U, V, W for these 49 terms.
            
            # Let's do it.
            
            # 2x2x2 tensors
            # U_2x2: 4x7 (indices 0..3 for A elements, 0..6 for terms)
            # A elements: 00, 01, 10, 11 -> indices 0, 1, 2, 3
            # U_2x2[t, idx]
            
            U_2x2 = [
                [1, 0, 1, 0], # T1: A00+A11
                [0, 0, 1, 1], # T2: A10+A11
                [1, 0, 0, 0], # T3: A00
                [0, 0, 0, 1], # T4: A11
                [1, 1, 0, 0], # T5: A00+A01
                [-1, 0, 1, 0],# T6: A10-A00
                [0, 1, 0, -1] # T7: A01-A11
            ]
            
            V_2x2 = [
                [1, 0, 0, 1], # T1: B00+B11
                [1, 0, 0, 0], # T2: B00
                [0, 1, 0, -1],# T3: B01-B11
                [0, 0, 1, -1],# T4: B10-B00
                [0, 0, 0, 1], # T5: B11
                [1, 1, 0, 0], # T6: B00+B01
                [0, 0, 1, 1]  # T7: B10+B11
            ]
            
            # W_2x2: 4x7 (indices 0..3 for C elements, 0..6 for terms)
            # C elements: 00, 01, 10, 11 -> indices 0, 1, 2, 3
            W_2x2 = [
                [1, 0, 0, 0, -1, 0, 1], # C00: T1+T4-T5+T7
                [0, 0, 1, 0, 1, 0, 0],  # C01: T3+T5
                [0, 1, 0, 1, 0, 0, 0],  # C10: T2+T4
                [1, -1, 1, 0, 0, 1, 0]  # C11: T1-T2+T3+T6
            ]
            
            # Macro steps coefficients
            # A_macro[s, b] where b is block index 0..3
            A_macro = [
                [1, 0, 0, 1], # M1
                [0, 0, 1, 1], # M2
                [1, 0, 0, 0], # M3
                [0, 0, 0, 1], # M4
                [1, 1, 0, 0], # M5
                [-1, 0, 1, 0],# M6
                [0, 1, 0, -1] # M7
            ]
            
            B_macro = [
                [1, 0, 0, 1], # M1
                [1, 0, 0, 0], # M2
                [0, 1, 0, -1],# M3
                [0, 0, 1, -1],# M4
                [0, 0, 0, 1], # M5
                [1, 1, 0, 0], # M6
                [0, 0, 1, 1]  # M7
            ]
            
            # C_macro[s, c] where c is block index 0..3
            C_macro = [
                [1, 0, 0, 1, -1, 0, 1], # C00
                [0, 0, 1, 0, 1, 0, 0],  # C01
                [0, 1, 0, 1, 0, 0, 0],  # C10
                [1, -1, 1, 0, 0, 1, 0]  # C11
            ]
            # Wait, C_macro is transposed relative to the loop.
            # We iterate s. For each s, we need the coefficient for each C block c.
            # C_coeff[s, c]
            C_coeff = [
                [1, 0, 0, 1], # s=0 (M1) contributes to C00(1), C11(1)
                [0, 0, 1, -1],# s=1 (M2) contributes to C10(1), C11(-1)
                [0, 1, 0, 1], # s=2 (M3) contributes to C01(1), C11(1)
                [1, 0, 1, 0], # s=3 (M4) contributes to C00(1), C10(1)
                [-1, 1, 0, 0],# s=4 (M5) contributes to C00(-1), C01(1)
                [0, 0, 0, 1], # s=5 (M6) contributes to C11(1)
                [1, 0, 0, 0]  # s=6 (M7) contributes to C00(1)
            ]
            
            # Now, for each s in 0..6, t in 0..6:
            for s in range(7):
                for t in range(7):
                    # Construct U_part, V_part, W_part for this term
                    # U_part is 16x1 (vector of coeffs for A)
                    # V_part is 16x1 (vector of coeffs for B)
                    # W_part is 16x1 (vector of coeffs for C)
                    
                    u_vec = [0] * 16
                    v_vec = [0] * 16
                    w_vec = [0] * 16
                    
                    # A coefficients
                    # u_vec[i*2+j] = sum_b A_macro[s,b] * U_2x2[t, b*2 + (i%2)*2 + (j%2)] ?
                    # Block b is at rows b//2, cols b%2.
                    # Element (r, c) in block b is at global row (b//2)*2 + r, col (b%2)*2 + c.
                    # Index in A: row*4 + col.
                    
                    for b in range(4):
                        br = b // 2
                        bc = b % 2
                        for r in range(2):
                            for c in range(2):
                                # Local index in 2x2 block
                                local_idx = r * 2 + c
                                # Global index in 4x4
                                global_r = br * 2 + r
                                global_c = bc * 2 + c
                                global_idx = global_r * 4 + global_c
                                
                                coeff = A_macro[s][b] * U_2x2[t][local_idx]
                                u_vec[global_idx] += coeff
                                
                                # B coefficients
                                # V_2x2[t][local_idx]
                                # Global index in B (4x4)
                                # Same mapping
                                v_vec[global_idx] += B_macro[s][b] * V_2x2[t][local_idx]
                    
                    # C coefficients
                    # w_vec[k*4+i]
                    # C block c is at rows c//2, cols c%2.
                    for c in range(4):
                        cr = c // 2
                        cc = c % 2
                        for r in range(2):
                            for c_local in range(2):
                                local_idx = r * 2 + c_local
                                global_r = cr * 2 + r
                                global_c = cc * 2 + c_local
                                global_idx = global_r * 4 + global_c
                                
                                coeff = C_coeff[s][c] * W_2x2[c][t]
                                w_vec[global_idx] += coeff
                    
                    # Add to U, V, W
                    for i in range(16):
                        if u_vec[i] != 0:
                            U[i][col] = u_vec[i]
                    for j in range(16):
                        if v_vec[j] != 0:
                            V[j][col] = v_vec[j]
                    for k in range(16):
                        if w_vec[k] != 0:
                            W[k][col] = w_vec[k]
                            
                    col += 1
                    
    # --- Part 2: A (4x4) * B2 (4x3) ---
    # Schoolbook
    # A is 4x4, B2 is 4x3.
    # C2 is 4x3.
    # C2[i, k] = sum_j A[i, j] B2[j, k]
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # U index: i*4 + j
                U[i*4 + j][col] = 1
                # V index: j*7 + (4+k)  (since B2 is columns 4,5,6)
                V[j*7 + (4+k)][col] = 1
                # W index: (4+k)*4 + i
                W[(4+k)*4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
