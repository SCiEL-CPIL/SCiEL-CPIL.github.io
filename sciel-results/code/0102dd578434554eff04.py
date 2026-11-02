import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split B into B4 (4x4) and B1 (4x1)
    # Part 1: A * B4 (4x4 * 4x4) using Strassen (Rank 49)
    # Part 2: A * B1 (4x4 * 4x1) using Schoolbook (Rank 16)
    
    # Helper: Strassen 2x2 decomposition
    # Returns U, V, W for 2x2x2 multiplication
    # A (2x2), B (2x2) -> C (2x2)
    # Rank 7
    def strassen_2x2():
        # Indices for A: 0,1,2,3 (row-major: a00, a01, a10, a11)
        # Indices for B: 0,1,2,3
        # Indices for C: 0,1,2,3
        
        # Strassen 2x2 formulas:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = []
        V = []
        W = []
        
        # M1
        U.append([1, 0, 1, 0]) # a00+a11
        V.append([1, 0, 0, 1]) # b00+b11
        W.append([1, 0, 0, 1]) # M1 contributes to C00 and C11
        
        # M2
        U.append([0, 0, 1, 1]) # a10+a11
        V.append([1, 0, 0, 0]) # b00
        W.append([0, 0, 1, -1]) # M2 contributes to C10 and -C11
        
        # M3
        U.append([1, 0, 0, 0]) # a00
        V.append([0, 1, 0, -1]) # b01-b11
        W.append([0, 1, 0, 1]) # M3 contributes to C01 and C11
        
        # M4
        U.append([0, 0, 0, 1]) # a11
        V.append([0, 0, 1, -1]) # b10-b00
        W.append([1, 0, 1, 0]) # M4 contributes to C00 and C10
        
        # M5
        U.append([1, 1, 0, 0]) # a00+a01
        V.append([0, 0, 0, 1]) # b11
        W.append([-1, 1, 0, 0]) # M5 contributes to -C00 and C01
        
        # M6
        U.append([0, 0, 1, -1]) # a10-a00
        V.append([1, 1, 0, 0]) # b00+b01
        W.append([0, 0, 0, 1]) # M6 contributes to C11
        
        # M7
        U.append([0, 1, 0, -1]) # a01-a11
        V.append([0, 0, 1, 1]) # b10+b11
        W.append([1, 0, 0, 0]) # M7 contributes to C00
        
        return U, V, W

    U_2x2, V_2x2, W_2x2 = strassen_2x2()
    
    # Strassen 4x4 decomposition
    # A is 4x4, B4 is 4x4.
    # We define 7 intermediate products Mk = Xk * Yk, where Xk, Yk are 2x2 matrices.
    # Xk is a linear combination of A's 2x2 blocks.
    # Yk is a linear combination of B4's 2x2 blocks.
    
    # A blocks: A00, A01, A10, A11
    # B4 blocks: B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the 16 elements of A to the 7 linear forms for Xk.
    # And 16 elements of B4 to the 7 linear forms for Yk.
    
    # Let's build U, V, W for the 4x4x4 part.
    # Total rank 49.
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    # For each of the 7 Strassen 4x4 steps:
    # We have a 2x2 matrix Xk and Yk.
    # We apply Strassen 2x2 to Xk*Yk.
    # This gives 7 scalar products.
    # We need to expand Xk and Yk in terms of A and B4 elements.
    
    # Helper to expand a 2x2 linear form into a 16-dim vector
    # Block indices: 00, 01, 10, 11
    # Element indices in 4x4 matrix (row-major):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    def expand_block_form(coeff_00, coeff_01, coeff_10, coeff_11):
        # Returns a list of 16 coefficients
        vec = [0] * 16
        # A00
        vec[0] = coeff_00
        vec[1] = coeff_00
        vec[4] = coeff_00
        vec[5] = coeff_00
        # A01
        vec[2] = coeff_01
        vec[3] = coeff_01
        vec[6] = coeff_01
        vec[7] = coeff_01
        # A10
        vec[8] = coeff_10
        vec[9] = coeff_10
        vec[12] = coeff_10
        vec[13] = coeff_10
        # A11
        vec[10] = coeff_11
        vec[11] = coeff_11
        vec[14] = coeff_11
        vec[15] = coeff_11
        return vec

    # Define the 7 Xk and Yk forms
    # Xk is a 2x2 matrix of coefficients for A blocks
    # Yk is a 2x2 matrix of coefficients for B4 blocks
    
    # M1: X1 = [[1,0],[0,1]], Y1 = [[1,0],[0,1]]
    # M2: X2 = [[0,0],[1,1]], Y2 = [[1,0],[0,0]]
    # M3: X3 = [[1,0],[0,0]], Y3 = [[0,1],[0,-1]]
    # M4: X4 = [[0,0],[0,1]], Y4 = [[0,0],[1,-1]]
    # M5: X5 = [[1,1],[0,0]], Y5 = [[0,0],[0,1]]
    # M6: X6 = [[0,0],[1,-1]], Y6 = [[1,1],[0,0]]
    # M7: X7 = [[0,1],[0,-1]], Y7 = [[0,0],[1,1]]
    
    forms_X = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,-1]],
        [[0,1],[0,-1]]
    ]
    
    forms_Y = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[0,0],[1,-1]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    
    # For each k in 0..6:
    for k in range(7):
        Xk = forms_X[k]
        Yk = forms_Y[k]
        
        # Xk is a 2x2 matrix of coefficients for A blocks.
        # We need to apply Strassen 2x2 to Xk * Yk.
        # Strassen 2x2 operates on 2x2 matrices of scalars.
        # Here Xk and Yk are "symbolic" 2x2 matrices.
        # We need to expand the Strassen 2x2 steps.
        
        # Strassen 2x2 uses 7 products:
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # Each Pi is a product of two linear forms.
        # We need to expand each Pi into scalar products of A and B4 elements.
        
        # Helper to add a product of two linear forms to U, V, W
        # The linear form for A is a 16-dim vector.
        # The linear form for B is a 16-dim vector.
        # The result contributes to the 2x2 matrix Xk*Yk.
        # The 2x2 matrix Xk*Yk contributes to the final 4x4 C.
        
        # We need to track how the 2x2 result maps to the 4x4 result.
        # For M1: C00 += M100 + M111, C01 += M101 + M110, etc?
        # No, M1 is a 2x2 matrix.
        # M1 = X1 Y1.
        # The result of X1 Y1 is added to C according to Strassen 4x4 formulas.
        
        # Let's define the contribution of each 2x2 element of Mk to the 4x4 C.
        # M1: C00 += M100, C01 += M101, C10 += M110, C11 += M111
        #      C00 += M111, C11 += M100 (from M1+M4-M5+M7? No)
        
        # Strassen 4x4 formulas:
        # C00 = M100 + M111 + M400 + M411 - M500 - M511 + M700 + M711
        # C01 = M101 + M110 + M301 + M310 + M501 + M510
        # C10 = M110 + M101 + M210 + M211 + M410 + M401
        # C11 = M111 + M100 - M211 + M210 + M311 + M300 + M611 + M600
        
        # This is getting complicated.
        # I will use a simpler approach:
        # Just use Schoolbook for the 4x4x4 part? No, that's 64.
        # Strassen is 49.
        
        # Given the complexity, I will fall back to a simpler but still improved method:
        # Use Winograd 2x2 tiling (Rank 56) for 4x4x4.
        # Total Rank 56 + 16 = 72.
        # This is worse than 65 but easier to implement?
        # Actually, 72 is worse than 80? No, 72 < 80.
        # But 65 is better.
        
        # I will implement the 72-rank solution (Tiling Winograd 2x2) as it is more robust to implement without errors in the linear form mapping.
        
    # Let's restart the implementation with Tiling Winograd 2x2.
    
    U = []
    V = []
    W = []
    
    # Winograd 2x2 decomposition
    # A (2x2), B (2x2) -> C (2x2)
    # Rank 7
    # U_2x2, V_2x2, W_2x2 from above
    
    # For 4x4x4:
    # We have 4 blocks in C: C00, C01, C10, C11.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # We decompose each product (e.g., A00 B00) using Winograd 2x2.
    # There are 8 such products.
    
    # For each product, we add 7 rank units.
    # Total 56 rank units.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A00: rows 0-1, cols 0-1.
    # A01: rows 0-1, cols 2-3.
    # A10: rows 2-3, cols 0-1.
    # A11: rows 2-3, cols 2-3.
    
    # B00: rows 0-1, cols 0-1.
    # B01: rows 0-1, cols 2-3.
    # B10: rows 2-3, cols 0-1.
    # B11: rows 2-3, cols 2-3.
    
    # C00: rows 0-1, cols 0-1.
    # C01: rows 0-1, cols 2-3.
    # C10: rows 2-3, cols 0-1.
    # C11: rows 2-3, cols 2-3.
    
    # For each of the 8 products:
    # 1. A_block B_block
    # 2. We get 7 scalar products.
    # 3. The result of A_block B_block is a 2x2 matrix.
    # 4. This 2x2 matrix is added to the corresponding C_block.
    
    # We need to expand the U, V, W vectors for each 2x2 product.
    
    products = [
        (0, 0, 0), # A00 B00 -> C00
        (0, 1, 1), # A00 B01 -> C01
        (1, 0, 0), # A01 B10 -> C00
        (1, 1, 1), # A01 B11 -> C01
        (2, 0, 2), # A10 B00 -> C10
        (2, 1, 3), # A10 B01 -> C11
        (3, 0, 2), # A11 B10 -> C10
        (3, 1, 3)  # A11 B11 -> C11
    ]
    
    # Block row/col offsets in 4x4 matrix
    # Block 0: 0,0
    # Block 1: 0,1
    # Block 2: 1,0
    # Block 3: 1,1
    
    def get_block_indices(block_id):
        r, c = divmod(block_id, 2)
        return r * 2, c * 2
    
    for a_blk, b_blk, c_blk in products:
        # Get 2x2 Winograd decomposition
        # We need to shift the indices for A and B.
        # A_block is at (r_a, c_a). B_block is at (r_b, c_b).
        # The 2x2 Winograd U, V, W are for local 2x2 indices.
        
        r_a, c_a = get_block_indices(a_blk)
        r_b, c_b = get_block_indices(b_blk)
        r_c, c_c = get_block_indices(c_blk)
        
        for i in range(7):
            # U vector for A
            u_vec = [0] * 16
            for j in range(4):
                # j is 0..3 for 2x2 matrix
                rr, cc = divmod(j, 2)
                # Global index
                g_r = r_a + rr
                g_c = c_a + cc
                g_idx = g_r * 4 + g_c
                u_vec[g_idx] = U_2x2[i][j]
            
            # V vector for B
            v_vec = [0] * 16
            for j in range(4):
                rr, cc = divmod(j, 2)
                g_r = r_b + rr
                g_c = c_b + cc
                g_idx = g_r * 4 + g_c
                v_vec[g_idx] = V_2x2[i][j]
            
            # W vector for C
            # The result of the 2x2 product is added to C_block.
            # The Winograd W_2x2[i] gives the contribution to the 2x2 result.
            w_vec = [0] * 16
            for j in range(4):
                rr, cc = divmod(j, 2)
                g_r = r_c + rr
                g_c = c_c + cc
                g_idx = g_r * 4 + g_c
                w_vec[g_idx] = W_2x2[i][j]
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
    
    # Now add the 4x4x1 part (Schoolbook)
    # A (4x4) * B1 (4x1) -> C1 (4x1)
    # C1_i = sum_k A_ik B1_k
    # Rank 16.
    
    for i in range(4):
        for k in range(4):
            u_vec = [0] * 16
            u_vec[i * 4 + k] = 1
            
            v_vec = [0] * 5 # B is 4x5, so B1 is the 5th column?
            # Wait, B is 4x5.
            # B4 is cols 0-3. B1 is col 4.
            # So V vector should be size 20 (4x5).
            # My previous U, V, W were for 4x4x4.
            # I need to extend them to 4x4x5.
            
            # Let's restart the U, V, W construction for 4x4x5.
            
    # RESTART: Construct U, V, W for 4x4x5 directly.
    
    U_final = []
    V_final = []
    W_final = []
    
    # Part 1: 4x4x4 using Tiling Winograd 2x2 (Rank 56)
    for a_blk, b_blk, c_blk in products:
        r_a, c_a = get_block_indices(a_blk)
        r_b, c_b = get_block_indices(b_blk)
        r_c, c_c = get_block_indices(c_blk)
        
        for i in range(7):
            u_vec = [0] * 16
            for j in range(4):
                rr, cc = divmod(j, 2)
                g_idx = (r_a + rr) * 4 + (c_a + cc)
                u_vec[g_idx] = U_2x2[i][j]
            
            v_vec = [0] * 20 # 4x5 matrix
            for j in range(4):
                rr, cc = divmod(j, 2)
                # B4 is cols 0-3. So c_b + cc is in 0-3.
                g_idx = (r_b + rr) * 5 + (c_b + cc)
                v_vec[g_idx] = V_2x2[i][j]
            
            w_vec = [0] * 20 # 5x4 matrix (p*n)
            # W is (p*n) x rank. p=5, n=4. Size 20.
            # C is 4x4. C_block is at (r_c, c_c).
            # W index: k*n + i.
            # k is row of C (0-3). i is col of C (0-3).
            # Wait, the contract says W[k*n+i, r].
            # C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            # So W index is k*n + i.
            # Here C is 4x4. i is row, k is col.
            # So W index is k*4 + i.
            
            for j in range(4):
                rr, cc = divmod(j, 2)
                # rr is row of C block (0-1). cc is col of C block (0-1).
                # Global row i = r_c + rr.
                # Global col k = c_c + cc.
                # W index = k * 4 + i.
                g_i = r_c + rr
                g_k = c_c + cc
                w_idx = g_k * 4 + g_i
                w_vec[w_idx] = W_2x2[i][j]
            
            U_final.append(u_vec)
            V_final.append(v_vec)
            W_final.append(w_vec)
    
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    # A (4x4) * B1 (4x1) -> C1 (4x1)
    # C1_i = sum_k A_ik B1_k
    # B1 is the 5th column of B (index 4).
    # C1 is the 5th column of C (index 4).
    
    for i in range(4):
        for k in range(4):
            u_vec = [0] * 16
            u_vec[i * 4 + k] = 1
            
            v_vec = [0] * 20
            v_vec[k * 5 + 4] = 1 # B[k, 4]
            
            w_vec = [0] * 20
            # C[i, 4]
            # W index = k * n + i? No.
            # Contract: W[k*n+i, r].
            # C[i, k] = ...
            # Here i is row, k is col.
            # So W index = k * 4 + i.
            # C1 is at col 4. So k=4.
            # W index = 4 * 4 + i = 16 + i.
            w_vec[16 + i] = 1
            
            U_final.append(u_vec)
            V_final.append(v_vec)
            W_final.append(w_vec)
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
