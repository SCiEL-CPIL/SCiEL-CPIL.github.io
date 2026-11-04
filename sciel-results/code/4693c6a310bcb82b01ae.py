import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index in U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define the 7 Strassen steps for 2x2 blocks
    # Each step k (0..6) defines X_k and Y_k as 2x2 matrices of linear combos of A and B blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We will represent linear combos as dicts: { (i,j): coeff }
    # where (i,j) is the index in the 4x4 matrix.
    
    # Block A00: (0,0), (0,1), (1,0), (1,1)
    # Block A01: (0,2), (0,3), (1,2), (1,3)
    # Block A10: (2,0), (2,1), (3,0), (3,1)
    # Block A11: (2,2), (2,3), (3,2), (3,3)
    
    # Similarly for B.
    
    # We define the 7 X_k and Y_k as lists of 4 dicts (for 2x2 entries)
    
    # Step 1: (A00 + A11) (B00 + B11)
    # X1_00 = A00_00 + A11_00, etc.
    
    # To make it easier, we define the block combinations first.
    # Block combo: dict of block indices to coeffs.
    # Then expand to entry indices.
    
    def expand_block_combo(block_combo, is_A):
        # block_combo: { (bi, bj): coeff }
        # is_A: True if A, False if B
        entries = {}
        for (bi, bj), coeff in block_combo.items():
            # bi, bj in {0,1}
            # Base row: 2*bi, Base col: 2*bj
            for di in range(2):
                for dj in range(2):
                    i = 2*bi + di
                    j = 2*bj + dj
                    entries[(i,j)] = entries.get((i,j), 0) + coeff
        return entries
    
    # Define the 7 steps
    # Each step is a tuple (X_blocks, Y_blocks)
    # X_blocks: list of 4 dicts (for X_00, X_01, X_10, X_11)
    # Y_blocks: list of 4 dicts (for Y_00, Y_01, Y_10, Y_11)
    
    steps = []
    
    # 1
    x1 = [
        {(0,0):1, (1,1):1}, # X00
        {(0,0):1, (1,1):1}, # X01 - Wait, X is 2x2 matrix.
        # X1 = A00 + A11. This is a 2x2 matrix where every entry is the sum of corresponding entries of A00 and A11.
        # So X1_00 = A00_00 + A11_00
        # X1_01 = A00_01 + A11_01
        # X1_10 = A00_10 + A11_10
        # X1_11 = A00_11 + A11_11
        # My expand_block_combo handles this if I pass the block combo.
        # But X1 is a single matrix.
        # I need to represent X1 as 4 dicts, one for each entry of the 2x2 matrix X1.
        # Entry (r,c) of X1 corresponds to entry (r,c) of A00 and A11.
    ]
    
    # Let's refine the representation.
    # For each of the 7 steps, we have a 2x2 matrix X and a 2x2 matrix Y.
    # Each entry of X is a linear combo of A entries.
    # Each entry of Y is a linear combo of B entries.
    
    # We will store for each step k:
    # X_entries: 2x2 list of dicts { (i,j): coeff }
    # Y_entries: 2x2 list of dicts { (i,j): coeff }
    
    def get_block_entries(block_i, block_j, is_A):
        # Returns 2x2 list of dicts for the entries of the block
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                i = 2*block_i + r
                j = 2*block_j + c
                res[r][c] = {(i,j): 1}
        return res
        
    def add_block_entries(A, B, coeff_A=1, coeff_B=1):
        # A, B are 2x2 lists of dicts
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                d = {}
                for k, v in A[r][c].items():
                    d[k] = d.get(k, 0) + v * coeff_A
                for k, v in B[r][c].items():
                    d[k] = d.get(k, 0) + v * coeff_B
                res[r][c] = d
        return res
        
    def sub_block_entries(A, B, coeff_A=1, coeff_B=1):
        return add_block_entries(A, B, coeff_A, -coeff_B)

    # Define blocks
    A00 = get_block_entries(0, 0, True)
    A01 = get_block_entries(0, 1, True)
    A10 = get_block_entries(1, 0, True)
    A11 = get_block_entries(1, 1, True)
    
    B00 = get_block_entries(0, 0, False)
    B01 = get_block_entries(0, 1, False)
    B10 = get_block_entries(1, 0, False)
    B11 = get_block_entries(1, 1, False)
    
    # Step 1: X = A00 + A11, Y = B00 + B11
    steps.append((add_block_entries(A00, A11), add_block_entries(B00, B11)))
    # Step 2: X = A10 + A11, Y = B00
    steps.append((add_block_entries(A10, A11), B00))
    # Step 3: X = A00, Y = B01 - B11
    steps.append((A00, sub_block_entries(B01, B11)))
    # Step 4: X = A11, Y = B10 - B00
    steps.append((A11, sub_block_entries(B10, B00)))
    # Step 5: X = A00 + A01, Y = B11
    steps.append((add_block_entries(A00, A01), B11))
    # Step 6: X = A10 - A00, Y = B00 + B01
    steps.append((sub_block_entries(A10, A00), add_block_entries(B00, B01)))
    # Step 7: X = A01 - A11, Y = B10 + B11
    steps.append((sub_block_entries(A01, A11), add_block_entries(B10, B11)))
    
    col = 0
    for k in range(7):
        X, Y = steps[k]
        # Apply Strassen to X, Y
        # X, Y are 2x2 lists of dicts.
        # We need the 7 scalar products.
        
        # S1 = (X00 + X11) (Y00 + Y11)
        # S2 = (X10 + X11) Y00
        # S3 = X00 (Y01 - Y11)
        # S4 = X11 (Y10 - Y00)
        # S5 = (X00 + X01) Y11
        # S6 = (X10 - X00) (Y00 + Y01)
        # S7 = (X01 - X11) (Y10 + Y11)
        
        # We will compute the U and V vectors for each of these 7 products.
        # And we will accumulate the W contributions.
        
        # To compute W, we need to know how each S contributes to C.
        # C = [ C00 C01; C10 C11 ] in block terms.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a 2x2 matrix.
        # M = X Y.
        # We need to decompose M = X Y into scalar products.
        # The scalar products are S1..S7.
        # We need the coefficients of S1..S7 in each entry of M.
        
        # Strassen reconstruction for 2x2:
        # M00 = S1 + S4 - S5 + S7
        # M01 = S3 + S5
        # M10 = S2 + S4
        # M11 = S1 - S2 + S3 + S6
        
        # So for step k, the contribution to C blocks is:
        # C00 += M_k00 + M_k10? No.
        # C_block_00 = M1_00 + M4_00 - M5_00 + M7_00 ... No.
        # C00 = M1 + M4 - M5 + M7.
        # This means C00_00 = M1_00 + M4_00 - M5_00 + M7_00.
        
        # So we can just add the coefficients.
        
        # For each of the 7 scalar products in step k:
        # We determine U_vec, V_vec.
        # We determine the contribution to C.
        
        # Let's define the 7 sub-products for step k
        sub_products = []
        # 1
        x1 = add_block_entries(X[0][0], X[1][1])
        y1 = add_block_entries(Y[0][0], Y[1][1])
        sub_products.append((x1, y1))
        # 2
        x2 = add_block_entries(X[1][0], X[1][1])
        y2 = Y[0][0]
        sub_products.append((x2, y2))
        # 3
        x3 = X[0][0]
        y3 = sub_block_entries(Y[0][1], Y[1][1])
        sub_products.append((x3, y3))
        # 4
        x4 = X[1][1]
        y4 = sub_block_entries(Y[1][0], Y[0][0])
        sub_products.append((x4, y4))
        # 5
        x5 = add_block_entries(X[0][0], X[0][1])
        y5 = Y[1][1]
        sub_products.append((x5, y5))
        # 6
        x6 = sub_block_entries(X[1][0], X[0][0])
        y6 = add_block_entries(Y[0][0], Y[0][1])
        sub_products.append((x6, y6))
        # 7
        x7 = sub_block_entries(X[0][1], X[1][1])
        y7 = add_block_entries(Y[1][0], Y[1][1])
        sub_products.append((x7, y7))
        
        for j in range(7):
            x_dict, y_dict = sub_products[j]
            
            # Fill U
            for (i, j_a), coeff in x_dict.items():
                U[u_idx(i, j_a)][col] += coeff
                
            # Fill V
            for (j_b, k), coeff in y_dict.items():
                V[v_idx(j_b, k)][col] += coeff
                
            # Fill W
            # We need to know which C entries this S_j contributes to.
            # S_j is the j-th Strassen product for M_k.
            # M_k = X Y.
            # The entries of M_k are linear combos of S_1..S_7.
            # M00 = S1 + S4 - S5 + S7
            # M01 = S3 + S5
            # M10 = S2 + S4
            # M11 = S1 - S2 + S3 + S6
            
            # C_block_00 = M1 + M4 - M5 + M7
            # C_block_01 = M3 + M5
            # C_block_10 = M2 + M4
            # C_block_11 = M1 - M2 + M3 + M6
            
            # So for step k, we add to C blocks based on k.
            # k=0 (M1):
            #   C00 += M1
            #   C11 += M1
            # k=1 (M2):
            #   C10 += M2
            #   C11 -= M2
            # k=2 (M3):
            #   C01 += M3
            #   C11 += M3
            # k=3 (M4):
            #   C00 += M4
            #   C10 += M4
            # k=4 (M5):
            #   C00 -= M5
            #   C01 += M5
            # k=5 (M6):
            #   C11 += M6
            # k=6 (M7):
            #   C00 += M7
            
            # We need the coefficients of S_j in M00, M01, M10, M11.
            # M00: [1, 0, 0, 1, -1, 0, 1] for S1..S7
            # M01: [0, 0, 1, 0, 1, 0, 0]
            # M10: [0, 1, 0, 1, 0, 0, 0]
            # M11: [1, -1, 1, 0, 0, 1, 0]
            
            coeffs_M = [
                [1, 0, 0, 1, -1, 0, 1], # M00
                [0, 0, 1, 0, 1, 0, 0], # M01
                [0, 1, 0, 1, 0, 0, 0], # M10
                [1, -1, 1, 0, 0, 1, 0]  # M11
            ]
            
            c_coeff = coeffs_M[j] # Wait, j is the index of S in 1..7.
            # coeffs_M[r][j] is the coefficient of S_{j+1} in M_r.
            
            # Now map M_r to C blocks.
            # k=0: C00 += M, C11 += M
            # k=1: C10 += M, C11 -= M
            # k=2: C01 += M, C11 += M
            # k=3: C00 += M, C10 += M
            # k=4: C00 -= M, C01 += M
            # k=5: C11 += M
            # k=6: C00 += M
            
            # For each M entry (r=0..3), we add c_coeff * block_coeff to C.
            
            # Block 00: rows 0,1 cols 0,1
            # Block 01: rows 0,1 cols 2,3
            # Block 10: rows 2,3 cols 0,1
            # Block 11: rows 2,3 cols 2,3
            
            block_coeffs = [0, 0, 0, 0] # for C00, C01, C10, C11
            
            if k == 0:
                block_coeffs = [1, 0, 0, 1]
            elif k == 1:
                block_coeffs = [0, 0, 1, -1]
            elif k == 2:
                block_coeffs = [0, 1, 0, 1]
            elif k == 3:
                block_coeffs = [1, 0, 1, 0]
            elif k == 4:
                block_coeffs = [-1, 1, 0, 0]
            elif k == 5:
                block_coeffs = [0, 0, 0, 1]
            elif k == 6:
                block_coeffs = [1, 0, 0, 0]
                
            # r=0 -> M00 -> C_block_00_00, C_block_00_01, etc?
            # No, M is a 2x2 matrix.
            # M00 contributes to C_block_r, col 0?
            # C_block is 2x2.
            # M00 is top-left of M.
            # So M00 contributes to C_block_row, col 0?
            # Yes.
            
            # So for each r in 0..3 (M entry index):
            # m_coeff = coeffs_M[r][j]
            # if m_coeff != 0:
            #   for each block b in 0..3 where block_coeffs[b] != 0:
            #     c_block_coeff = block_coeffs[b] * m_coeff
            #     Add c_block_coeff to W for the corresponding C entry.
            
            for r in range(4):
                m_coeff = coeffs_M[r][j]
                if m_coeff == 0: continue
                
                # r=0: M00 (top-left)
                # r=1: M01 (top-right)
                # r=2: M10 (bottom-left)
                # r=3: M11 (bottom-right)
                
                # Determine the row and col within the 2x2 block
                if r == 0:
                    dr, dc = 0, 0
                elif r == 1:
                    dr, dc = 0, 1
                elif r == 2:
                    dr, dc = 1, 0
                else:
                    dr, dc = 1, 1
                    
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    
                    # Block b:
                    # b=0: rows 0,1 cols 0,1
                    # b=1: rows 0,1 cols 2,3
                    # b=2: rows 2,3 cols 0,1
                    # b=3: rows 2,3 cols 2,3
                    
                    base_row = 2 * (b // 2)
                    base_col = 2 * (b % 2)
                    
                    i = base_row + dr
                    k_c = base_col + dc
                    
                    W[w_idx(k_c, i)][col] += bc * m_coeff
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
