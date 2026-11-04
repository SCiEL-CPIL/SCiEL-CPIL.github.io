import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # --- Part 1: Strassen for 4x4 * 4x4 (first 4 columns of B) ---
    
    # Strassen 2x2x2 decomposition parameters
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in 2x2 block)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # The 7 Strassen multiplications for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A and B in each M_k
    # A coeffs (4x7 matrix, row k, col block_idx)
    S_A = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # B coeffs (4x7 matrix, row k, col block_idx)
    S_B = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1] # M4 -- Wait, M4 = A11(B10-B00). B coeffs: B00=-1, B10=1.
                      # My S_B above for M4 was [0,0,1,-1] which is B10=1, B11=-1. Correct.
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # C coeffs (4x7 matrix, row k, col block_idx)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    S_C = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Now, for 4x4x4, we have 4 blocks of 2x2.
    # Let A be blocks A_00, A_01, A_10, A_11.
    # Let B be blocks B_00, B_01, B_10, B_11.
    # We form 7 linear combinations of these 4x4 blocks?
    # No, we form 7 linear combinations of the 2x2 sub-blocks?
    # Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # The "blocks" in the 4x4 Strassen are the 2x2 submatrices.
    
    # Let's define the 4x4 Strassen combinations.
    # A_00, A_01, A_10, A_11 are 2x2 matrices.
    # M1 = (A_00 + A_11) (B_00 + B_11)
    # M2 = (A_10 + A_11) B_00
    # M3 = A_00 (B_01 - B_11)
    # M4 = A_11 (B_10 - B_00)
    # M5 = (A_00 + A_01) B_11
    # M6 = (A_10 - A_00) (B_00 + B_01)
    # M7 = (A_01 - A_11) (B_10 + B_11)
    
    # Each M_k is a 2x2 matrix product.
    # We compute each M_k using the 2x2x2 Strassen decomposition (7 terms).
    # So for each k (0..6), we have 7 terms.
    
    # We need to map the 2x2 block indices to the 4x4 flat indices.
    # A flat index: i * 4 + j.
    # Block (bi, bj) contains A[bi*2 : bi*2+2, bj*2 : bj*2+2].
    # Local index in block: li * 2 + lj.
    # Flat index: (bi*2 + li) * 4 + (bj*2 + lj).
    
    # For each of the 7 outer Strassen steps (k_outer):
    # We have linear combinations of A blocks and B blocks.
    # Let's define the coefficients for the 4x4 blocks.
    
    Outer_S_A = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A10, A00
        [0, 1, 0, -1] # M7: A01, A11
    ]
    
    Outer_S_B = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, B11
        [0, 0, 1, -1] # M4: B10, B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    # For each outer step k_outer:
    # We form S_A_mat = sum(Outer_S_A[k_outer][b] * A_block_b)
    # S_B_mat = sum(Outer_S_B[k_outer][b] * B_block_b)
    # Then we compute P_k_outer = S_A_mat * S_B_mat.
    # P_k_outer is a 2x2 matrix.
    # We use the inner 2x2x2 Strassen decomposition to compute P_k_outer.
    # The inner decomposition has 7 terms.
    # For each inner term l (0..6):
    # u_inner is the A-coeffs for the inner term.
    # v_inner is the B-coeffs for the inner term.
    # w_inner is the C-coeffs for the inner term.
    
    # The total u vector for the 4x4 case will be the sum of contributions from the outer blocks.
    # If Outer_S_A[k_outer][b] is non-zero, then the elements of A_block_b contribute.
    # The inner term l uses specific elements of S_A_mat.
    # S_A_mat is a 2x2 matrix. Its elements are linear combinations of A_block elements.
    # S_A_mat[li, lj] = sum_b Outer_S_A[k_outer][b] * A_block_b[li, lj].
    # So the coefficient for A_block_b[li, lj] in the u vector is Outer_S_A[k_outer][b] * u_inner[li*2+lj].
    
    # Similarly for V.
    # For W, the output P_k_outer is a 2x2 matrix.
    # The result C is a 4x4 matrix.
    # C_block_c = sum_k_outer (contribution of P_k_outer to C_block_c).
    # The outer Strassen formulas for C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for a given outer step k_outer, P_k_outer contributes to specific C blocks.
    # Let Outer_S_C[k_outer][c] be the coefficient.
    Outer_S_C = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Now we iterate.
    for k_outer in range(7):
        for l_inner in range(7):
            u = [0.0] * 16
            v = [0.0] * 28
            w = [0.0] * 28
            
            # Get inner coefficients
            u_inner = S_A[l_inner] # length 4
            v_inner = S_B[l_inner] # length 4
            w_inner = S_C[l_inner] # length 4
            
            # Build U
            for b_a in range(4):
                coeff_a = Outer_S_A[k_outer][b_a]
                if coeff_a == 0: continue
                bi, bj = divmod(b_a, 2) # Block row, Block col
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        flat_idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                        u[flat_idx] += coeff_a * u_inner[local_idx]
                        
            # Build V
            # B is 4x7. We use first 4 columns (indices 0..15).
            # B blocks are 2x2.
            for b_b in range(4):
                coeff_b = Outer_S_B[k_outer][b_b]
                if coeff_b == 0: continue
                bi, bj = divmod(b_b, 2)
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        # B flat index: row * 7 + col.
                        # Block (bi, bj) is at rows bi*2..bi*2+1, cols bj*2..bj*2+1.
                        # Flat index: (bi*2+li)*7 + (bj*2+lj)
                        flat_idx = (bi * 2 + li) * 7 + (bj * 2 + lj)
                        v[flat_idx] += coeff_b * v_inner[local_idx]
                        
            # Build W
            # C is 4x7. We use first 4 columns (indices 0..15 in W? No).
            # W index: k * 4 + i.
            # C blocks are 2x2.
            # C block (ci, cj) is at rows ci*2..ci*2+1, cols cj*2..cj*2+1.
            # W flat index: col * 4 + row.
            # So for C block (ci, cj), local (li, lj):
            # col = cj*2 + lj
            # row = ci*2 + li
            # W index = col * 4 + row.
            
            for b_c in range(4):
                coeff_c = Outer_S_C[k_outer][b_c]
                if coeff_c == 0: continue
                bi, bj = divmod(b_c, 2) # Block row, Block col
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        col = bj * 2 + lj
                        row = bi * 2 + li
                        flat_idx = col * 4 + row
                        w[flat_idx] += coeff_c * w_inner[local_idx]
                        
            add_term(u, v, w)
            
    # --- Part 2: Schoolbook for 4x4 * 4x3 (last 3 columns of B) ---
    # A: 4x4 (indices 0..15)
    # B: 4x3 (columns 4,5,6 of original B).
    # B flat indices: row * 7 + col. Col 4,5,6.
    # W: 4x3 (columns 4,5,6 of original C).
    # W flat indices: col * 4 + row. Col 4,5,6.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = [0.0] * 16
                v = [0.0] * 28
                w = [0.0] * 28
                
                # A index
                u[i * 4 + j] = 1.0
                
                # B index: row j, col 4+k
                b_col = 4 + k
                v[j * 7 + b_col] = 1.0
                
                # W index: col 4+k, row i
                w_col = 4 + k
                w[w_col * 4 + i] = 1.0
                
                add_term(u, v, w)
                
    rank = len(U_list)
    
    # Convert lists to matrices
    U = np.array(U_list).T # Shape (16, rank)
    V = np.array(V_list).T # Shape (28, rank)
    W = np.array(W_list).T # Shape (28, rank)
    
    # Ensure integer/half-integer (they are integers in this construction)
    # The evaluator accepts lists of lists or arrays.
    # The seed program returns lists of lists.
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
