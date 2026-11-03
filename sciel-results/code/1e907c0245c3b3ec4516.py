import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    # U, V, W are 4x7, 4x7, 4x7
    # Indices for 2x2: i,j in {0,1} -> idx = i*2+j
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00 idx 0, A11 idx 3
    # B00 idx 0, B11 idx 3
    # C00 idx 0, C11 idx 3
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1; W_22[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A10 idx 2, A11 idx 3
    # B00 idx 0
    # C10 idx 2, C11 idx 3
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[2, 1] = 1; W_22[3, 1] = 1
    
    # M3 = A00(B10-B11)
    # A00 idx 0
    # B10 idx 1, B11 idx 3
    # C00 idx 0, C01 idx 1
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[0, 2] = 1; W_22[1, 2] = 1
    
    # M4 = A11(B01-B00)
    # A11 idx 3
    # B01 idx 2, B00 idx 0
    # C10 idx 2, C11 idx 3
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    W_22[2, 3] = 1; W_22[3, 3] = 1
    
    # M5 = (A00+A01)B11
    # A00 idx 0, A01 idx 1
    # B11 idx 3
    # C00 idx 0, C01 idx 1
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[0, 4] = 1; W_22[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10 idx 2, A00 idx 0
    # B00 idx 0, B01 idx 2
    # C10 idx 2, C11 idx 3
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[2, 5] = 1
    W_22[2, 5] = 1; W_22[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01 idx 1, A11 idx 3
    # B10 idx 1, B11 idx 3
    # C00 idx 0, C01 idx 1
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[1, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1; W_22[1, 6] = 1
    
    # Combine M1..M7 to get C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express C as sum of rank-1 terms.
    # The W matrix above gives the contribution to C from each Mk.
    # But we need to combine them.
    # Actually, the W matrix I built above is for the "internal" 2x2 multiplication.
    # For the 4x4 case, we have 7 such multiplications.
    # Each multiplication produces a 2x2 matrix.
    # The final C is a 4x4 matrix composed of these 2x2 blocks.
    # So we just need to place the results of M1..M7 into the correct blocks.
    # But the Strassen formulas combine M1..M7 to form the blocks.
    # So we need to sum the contributions.
    
    # Let's build the 4x4x4 decomposition directly.
    # U_44 will be 16x49, V_44 16x49, W_44 16x49.
    # Each of the 7 Strassen steps uses 7 rank-1 terms.
    # Total 49 terms.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Block indices for 4x4:
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # For each Strassen step k (0..6), we have 7 rank-1 terms.
    # The U and V matrices for the k-th step are linear combinations of the blocks.
    # The W matrix places the result into the C blocks.
    
    # Step 0: M1 = (A00+A11)(B00+B11)
    # U: A00 (0-3) + A11 (12-15)
    # V: B00 (0-3) + B11 (12-15)
    # W: C00 (0-3) + C11 (12-15)
    for t in range(7):
        col = 0 * 7 + t
        # U
        U_44[0:4, col] = U_22[:, t]
        U_44[12:16, col] = U_22[:, t]
        # V
        V_44[0:4, col] = V_22[:, t]
        V_44[12:16, col] = V_22[:, t]
        # W
        W_44[0:4, col] = W_22[:, t]
        W_44[12:16, col] = W_22[:, t]
        
    # Step 1: M2 = (A10+A11)B00
    # U: A10 (8-11) + A11 (12-15)
    # V: B00 (0-3)
    # W: C10 (8-11) + C11 (12-15)
    for t in range(7):
        col = 1 * 7 + t
        U_44[8:12, col] = U_22[:, t]
        U_44[12:16, col] = U_22[:, t]
        V_44[0:4, col] = V_22[:, t]
        W_44[8:12, col] = W_22[:, t]
        W_44[12:16, col] = W_22[:, t]
        
    # Step 2: M3 = A00(B10-B11)
    # U: A00 (0-3)
    # V: B10 (8-11) - B11 (12-15)
    # W: C00 (0-3) + C01 (4-7)
    for t in range(7):
        col = 2 * 7 + t
        U_44[0:4, col] = U_22[:, t]
        V_44[8:12, col] = V_22[:, t]
        V_44[12:16, col] = -V_22[:, t]
        W_44[0:4, col] = W_22[:, t]
        W_44[4:8, col] = W_22[:, t]
        
    # Step 3: M4 = A11(B01-B00)
    # U: A11 (12-15)
    # V: B01 (4-7) - B00 (0-3)
    # W: C10 (8-11) + C11 (12-15)
    for t in range(7):
        col = 3 * 7 + t
        U_44[12:16, col] = U_22[:, t]
        V_44[4:8, col] = V_22[:, t]
        V_44[0:4, col] = -V_22[:, t]
        W_44[8:12, col] = W_22[:, t]
        W_44[12:16, col] = W_22[:, t]
        
    # Step 4: M5 = (A00+A01)B11
    # U: A00 (0-3) + A01 (4-7)
    # V: B11 (12-15)
    # W: C00 (0-3) + C01 (4-7)
    for t in range(7):
        col = 4 * 7 + t
        U_44[0:4, col] = U_22[:, t]
        U_44[4:8, col] = U_22[:, t]
        V_44[12:16, col] = V_22[:, t]
        W_44[0:4, col] = W_22[:, t]
        W_44[4:8, col] = W_22[:, t]
        
    # Step 5: M6 = (A10-A00)(B00+B01)
    # U: A10 (8-11) - A00 (0-3)
    # V: B00 (0-3) + B01 (4-7)
    # W: C10 (8-11) + C11 (12-15)
    for t in range(7):
        col = 5 * 7 + t
        U_44[8:12, col] = U_22[:, t]
        U_44[0:4, col] = -U_22[:, t]
        V_44[0:4, col] = V_22[:, t]
        V_44[4:8, col] = V_22[:, t]
        W_44[8:12, col] = W_22[:, t]
        W_44[12:16, col] = W_22[:, t]
        
    # Step 6: M7 = (A01-A11)(B10+B11)
    # U: A01 (4-7) - A11 (12-15)
    # V: B10 (8-11) + B11 (12-15)
    # W: C00 (0-3) + C01 (4-7)
    for t in range(7):
        col = 6 * 7 + t
        U_44[4:8, col] = U_22[:, t]
        U_44[12:16, col] = -U_22[:, t]
        V_44[8:12, col] = V_22[:, t]
        V_44[12:16, col] = V_22[:, t]
        W_44[0:4, col] = W_22[:, t]
        W_44[4:8, col] = W_22[:, t]
        
    # Now we have the 4x4x4 decomposition of rank 49.
    # We need to add the 4x4x5 part.
    # 4x4x5 = 4x4x4 (rank 49) + 4x4x1 (rank 16).
    # Total rank = 49 + 49 + 16 = 114.
    
    # 4x4x1 decomposition:
    # C_i = sum_j A_ij B_j
    # U: A_ij, V: B_j, W: C_i
    # Rank 16.
    U_41 = np.zeros((16, 16))
    V_41 = np.zeros((16, 16))
    W_41 = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_41[i * 4 + j, col] = 1
            V_41[j, col] = 1
            W_41[i * 4, col] = 1 # Wait, C is 4x1, so index is i.
            # But our C is 4x9. The 1st column of C.
            # So W_41[i, col] = 1.
            # But W_41 shape is 16x16? No, C is 4x1, so W has 4 rows.
            # But we are building for 4x4x9.
            # The 4x4x1 part corresponds to the first column of C.
            # So W_41 should be 4x16?
            # But we are concatenating with 4x4x4 which has W of size 16x49.
            # So we need to align the dimensions.
            # U is 16xR, V is 16xR, W is 16xR?
            # No, U is n*m x R, V is m*p x R, W is p*n x R.
            # n=4, m=4, p=9.
            # U: 16 x R
            # V: 36 x R
            # W: 36 x R
            # My 4x4x4 decomposition:
            # U: 16 x 49
            # V: 16 x 49 (for 4x4)
            # W: 16 x 49 (for 4x4)
            # But V should be 4x4=16.
            # For 4x4x9, V is 4x9=36.
            # So I need to embed the 4x4x4 V into the 4x9 V.
            # The first 4 columns of B are used for 4x4x4.
            # The next 4 columns for the second 4x4x4.
            # The last 1 column for 4x4x1.
            
    # Let's reconstruct the full U, V, W.
    R = 49 + 49 + 16
    U_full = np.zeros((16, R))
    V_full = np.zeros((36, R))
    W_full = np.zeros((36, R))
    
    # Part 1: 4x4x4 for columns 0-3 of B
    # U: 16x49
    # V: 16x49 (rows 0-15 of V_full)
    # W: 16x49 (rows 0-15 of W_full)
    U_full[:, 0:49] = U_44
    V_full[0:16, 0:49] = V_44
    W_full[0:16, 0:49] = W_44
    
    # Part 2: 4x4x4 for columns 4-7 of B
    # U: same U_44
    # V: 16x49 (rows 16-31 of V_full)
    # W: 16x49 (rows 16-31 of W_full)
    U_full[:, 49:98] = U_44
    V_full[16:32, 49:98] = V_44
    W_full[16:32, 49:98] = W_44
    
    # Part 3: 4x4x1 for column 8 of B
    # U: 16x16
    # V: 16x16 (rows 32-35 of V_full)
    # W: 16x16 (rows 32-35 of W_full)
    # 4x4x1 decomposition:
    # C_i = sum_j A_ij B_j
    # U_ij = 1, V_j = 1, W_i = 1
    for i in range(4):
        for j in range(4):
            col = 98 + i * 4 + j
            U_full[i * 4 + j, col] = 1
            V_full[32 + j, col] = 1
            W_full[32 + i, col] = 1
            
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

