def construct():
    n, m, p = 4, 4, 9
    
    # --- Strassen 4x4 x 4x4 ---
    U_S = [[0] * 49 for _ in range(16)]
    V_S = [[0] * 49 for _ in range(16)]
    W_S = [[0] * 49 for _ in range(16)]
    
    # 2x2 Strassen matrices
    U_2 = [
        [1, 0, 1, 0, 1, -1, 0], # P00
        [0, 0, 0, 0, 1, 0, 1],  # P01
        [0, 1, 0, 0, 0, 1, 0],  # P10
        [1, 1, 0, 1, 0, 0, -1]  # P11
    ]
    V_2 = [
        [1, 1, 0, -1, 0, 1, 0], # Q00
        [0, 0, 1, 0, 0, 1, 0],  # Q01
        [0, 0, 0, 1, 0, 0, 1],  # Q10
        [1, 0, -1, 0, 1, 0, 1]  # Q11
    ]
    W_2 = [
        [1, 0, 0, 1, -1, 0, 1], # R00
        [0, 0, 1, 0, 1, 0, 0],  # R01
        [0, 1, 0, 1, 0, 0, 0],  # R10
        [1, -1, 1, 0, 0, 1, 0]  # R11
    ]
    
    # Define A block indices
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Define B block indices (first 4 cols)
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # Define C block indices
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Strassen M blocks
    # M1: P=A00+A11, Q=B00+B11
    # M2: P=A10+A11, Q=B00
    # M3: P=A00, Q=B01-B11
    # M4: P=A11, Q=B10-B00
    # M5: P=A00+A01, Q=B11
    # M6: P=A10-A00, Q=B00+B01
    # M7: P=A01-A11, Q=B10+B11
    
    # For each M_k, we have 7 sub-multiplications.
    # col_base = k * 7
    
    # Helper to add U_2 contribution
    def add_U(U_S, U_2, p_blocks, col_base):
        # p_blocks is a dict: {0: [coeff, indices], 1: ..., 2: ..., 3: ...}
        # Or list of (coeff, block_idx)
        # U_2 row r corresponds to P_r (0:P00, 1:P01, 2:P10, 3:P11)
        for r in range(4):
            for sub_col in range(7):
                val = U_2[r][sub_col]
                if val != 0:
                    # Get the A indices for P_r
                    # This depends on the specific M_k
                    pass
                    
    # This is getting messy. I will hardcode the additions.
    
    # M1 (k=0): P=A00+A11, Q=B00+B11
    # P00 = A00[0]+A11[0] (idx 0, 10)
    # P01 = A00[1]+A11[1] (idx 1, 11)
    # P10 = A00[2]+A11[2] (idx 4, 14)
    # P11 = A00[3]+A11[3] (idx 5, 15)
    
    # Q00 = B00[0]+B11[0] (idx 0, 10)
    # Q01 = B00[1]+B11[1] (idx 1, 11)
    # Q10 = B00[2]+B11[2] (idx 4, 14)
    # Q11 = B00[3]+B11[3] (idx 5, 15)
    
    # Add to U_S, V_S
    for r in range(4):
        for sc in range(7):
            val = U_2[r][sc]
            if val:
                col = sc
                # P_r indices
                if r == 0: idxs = [0, 10]
                elif r == 1: idxs = [1, 11]
                elif r == 2: idxs = [4, 14]
                else: idxs = [5, 15]
                for idx in idxs:
                    U_S[idx][col] += val
                    
                # Q_r indices
                if r == 0: idxs = [0, 10]
                elif r == 1: idxs = [1, 11]
                elif r == 2: idxs = [4, 14]
                else: idxs = [5, 15]
                for idx in idxs:
                    V_S[idx][col] += val
                    
    # Add to W_S
    # C00 = M1...
    # C00 indices: 0,1,4,5
    # C01 indices: 2,3,6,7
    # C10 indices: 8,9,12,13
    # C11 indices: 10,11,14,15
    
    # W_2 row c (0:R00, 1:R01, 2:R10, 3:R11)
    # R00 -> C00
    # R01 -> C01
    # R10 -> C10
    # R11 -> C11
    
    for c in range(4):
        for sc in range(7):
            val = W_2[c][sc]
            if val:
                col = sc
                if c == 0: idxs = [0, 1, 4, 5]
                elif c == 1: idxs = [2, 3, 6, 7]
                elif c == 2: idxs = [8, 9, 12, 13]
                else: idxs = [10, 11, 14, 15]
                for idx in idxs:
                    W_S[idx][col] += val
