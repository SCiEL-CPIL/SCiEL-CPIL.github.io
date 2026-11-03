import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Winograd 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[c00, c01], [c10, c11]]
    # m1 = (a+d)(e+h) -> u=[1,0,1,0], v=[1,1,0,0], w=[1,0,0,1]
    # m2 = (c+d)e    -> u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # m3 = a(f-h)    -> u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # m4 = d(g-e)    -> u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
    # m5 = (a+b)h    -> u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
    # m6 = (c-a)(e+f) -> u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # m7 = (b-d)(g+h) -> u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    wino_U = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    wino_V = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    wino_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    # Strassen 4x4x4 decomposition
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to construct U, V, W for 4x4x4
    # U is 16x49, V is 16x49, W is 16x49
    # Indices for A: 0..3 (A11), 4..7 (A12), 8..11 (A21), 12..15 (A22)
    # Indices for B: 0..3 (B11), 4..7 (B12), 8..11 (B21), 12..15 (B22)
    # Indices for C: 0..3 (C11), 4..7 (C12), 8..11 (C21), 12..15 (C22)
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Helper to add a winograd block
    def add_wino_block(U, V, W, a_idx, b_idx, c_idx, coeff):
        # a_idx, b_idx, c_idx are lists of 4 indices for the 2x2 blocks
        # coeff is a scalar or vector? No, the winograd decomposition is linear.
        # The winograd U, V, W are for the 2x2 blocks.
        # We need to map them to the 4x4 indices.
        
        for r in range(7):
            for i in range(4):
                if wino_U[i, r] != 0:
                    U[a_idx[i], r] += wino_U[i, r] * coeff
                if wino_V[i, r] != 0:
                    V[b_idx[i], r] += wino_V[i, r] * coeff
                if wino_W[i, r] != 0:
                    W[c_idx[i], r] += wino_W[i, r] * coeff
                    
    # This is not quite right. The winograd decomposition computes M = A_block * B_block.
    # Then C = sum of M's.
    # So we need to add the contributions of each M to the final C.
    
    # Let's do it step by step.
    # M1 = (A11+A22)(B11+B22)
    # A_block = A11+A22, B_block = B11+B22
    # This is a 2x2x2 multiplication.
    # We can use the winograd decomposition for this.
    # But the inputs are sums.
    
    # Actually, it's easier to just construct the full U, V, W by expanding the Strassen formulas.
    
    # Let's define the Strassen coefficients.
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # For each M, we have a 2x2x2 multiplication.
    # Let's call the winograd matrices U_w, V_w, W_w.
    # For M1, the A_block is A11+A22.
    # So the U part for M1 is U_w * (A11+A22).
    # This means we add U_w to the A11 indices and U_w to the A22 indices.
    
    # Let's define the index mappings.
    # A11: 0,1,2,3
    # A12: 4,5,6,7
    # A21: 8,9,10,11
    # A22: 12,13,14,15
    
    # B11: 0,1,2,3
    # B12: 4,5,6,7
    # B21: 8,9,10,11
    # B22: 12,13,14,15
    
    # C11: 0,1,2,3
    # C12: 4,5,6,7
    # C21: 8,9,10,11
    # C22: 12,13,14,15
    
    # We will create 7 sets of 7 columns, total 49 columns.
    
    for m_idx in range(7):
        col_offset = m_idx * 7
        for r in range(7):
            col = col_offset + r
            
            # Determine A_block and B_block for this M
            if m_idx == 0: # M1
                a_blocks = [[0,1,2,3], [12,13,14,15]] # A11, A22
                b_blocks = [[0,1,2,3], [12,13,14,15]] # B11, B22
                coeffs_a = [1, 1]
                coeffs_b = [1, 1]
            elif m_idx == 1: # M2
                a_blocks = [[8,9,10,11], [12,13,14,15]] # A21, A22
                b_blocks = [[0,1,2,3]] # B11
                coeffs_a = [1, 1]
                coeffs_b = [1]
            elif m_idx == 2: # M3
                a_blocks = [[0,1,2,3]] # A11
                b_blocks = [[4,5,6,7], [12,13,14,15]] # B12, B22
                coeffs_a = [1]
                coeffs_b = [1, -1]
            elif m_idx == 3: # M4
                a_blocks = [[12,13,14,15]] # A22
                b_blocks = [[8,9,10,11], [0,1,2,3]] # B21, B11
                coeffs_a = [1]
                coeffs_b = [1, -1]
            elif m_idx == 4: # M5
                a_blocks = [[0,1,2,3], [4,5,6,7]] # A11, A12
                b_blocks = [[12,13,14,15]] # B22
                coeffs_a = [1, 1]
                coeffs_b = [1]
            elif m_idx == 5: # M6
                a_blocks = [[8,9,10,11], [0,1,2,3]] # A21, A11
                b_blocks = [[0,1,2,3], [4,5,6,7]] # B11, B12
                coeffs_a = [1, -1]
                coeffs_b = [1, 1]
            else: # M7
                a_blocks = [[4,5,6,7], [12,13,14,15]] # A12, A22
                b_blocks = [[8,9,10,11], [12,13,14,15]] # B21, B22
                coeffs_a = [1, -1]
                coeffs_b = [1, 1]
            
            # Now add the winograd contribution
            for i in range(4):
                # U part
                u_val = 0
                for k, block_idx in enumerate(a_blocks):
                    if i < 4: # wino_U is 4x7
                        u_val += wino_U[i, r] * coeffs_a[k]
                if u_val != 0:
                    # We need to add this to the correct row in U
                    # The row index in U corresponds to the flattened A
                    # But wino_U[i, r] is for the i-th element of the 2x2 block.
                    # We need to map i to the actual index in A.
                    # This is tricky because a_blocks can have multiple blocks.
                    # Actually, the winograd U is for a single 2x2 block.
                    # If A_block is a sum of two 2x2 blocks, the U part is the sum of the U parts for each block.
                    
                    # Let's restructure.
                    # For each block in a_blocks, add the winograd U contribution.
                    pass
            
            # This is getting complicated. Let's use a simpler approach.
            # Just expand the Strassen formulas directly.
            
    # Let's restart the 4x4x4 construction with a direct expansion.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 is a 2x2 matrix.
    # Let's define the winograd matrices for the 2x2 blocks.
    
    # We will use a helper function to add a winograd block contribution.
    def add_wino(U, V, W, a_idx, b_idx, c_idx, col, coeff=1.0):
        # a_idx, b_idx, c_idx are lists of 4 indices for the 2x2 blocks.
        # col is the column index in U, V, W.
        for i in range(4):
            U[a_idx[i], col] += wino_U[i, r] * coeff
            V[b_idx[i], col] += wino_V[i, r] * coeff
            W[c_idx[i], col] += wino_W[i, r] * coeff

    # This is still not quite right because the winograd decomposition is for a single 2x2x2 multiplication.
    # If A_block is A11+A22, then the U part is U_w for A11 plus U_w for A22.
    
    # Let's just hardcode the 49-rank decomposition.
    # Or, use the 56-rank decomposition (8 Winograd blocks) which is easier.
    # 56 + 8 = 64.
    # 64 is better than 80.
    
    # Let's implement the 56-rank decomposition.
    # 8 Winograd blocks for 4x4x4.
    # 8 rank for 4x4x1.
    
    U = np.zeros((16, 64))
    V = np.zeros((20, 64))
    W = np.zeros((20, 64))
    
    # 8 Winograd blocks for 4x4x4
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Block 1: A00 B00 -> C00
    # A00: 0,1,2,3; B00: 0,1,2,3; C00: 0,1,2,3
    for r in range(7):
        col = r
        U[0:4, col] = wino_U[:, r]
        V[0:4, col] = wino_V[:, r]
        W[0:4, col] = wino_W[:, r]
        
    # Block 2: A01 B10 -> C00
    # A01: 4,5,6,7; B10: 8,9,10,11; C00: 0,1,2,3
    for r in range(7):
        col = 7 + r
        U[4:8, col] = wino_U[:, r]
        V[8:12, col] = wino_V[:, r]
        W[0:4, col] = wino_W[:, r]
        
    # Block 3: A00 B01 -> C01
    # A00: 0,1,2,3; B01: 4,5,6,7; C01: 4,5,6,7
    for r in range(7):
        col = 14 + r
        U[0:4, col] = wino_U[:, r]
        V[4:8, col] = wino_V[:, r]
        W[4:8, col] = wino_W[:, r]
        
    # Block 4: A01 B11 -> C01
    # A01: 4,5,6,7; B11: 12,13,14,15; C01: 4,5,6,7
    for r in range(7):
        col = 21 + r
        U[4:8, col] = wino_U[:, r]
        V[12:16, col] = wino_V[:, r]
        W[4:8, col] = wino_W[:, r]
        
    # Block 5: A10 B00 -> C10
    # A10: 8,9,10,11; B00: 0,1,2,3; C10: 8,9,10,11
    for r in range(7):
        col = 28 + r
        U[8:12, col] = wino_U[:, r]
        V[0:4, col] = wino_V[:, r]
        W[8:12, col] = wino_W[:, r]
        
    # Block 6: A11 B10 -> C10
    # A11: 12,13,14,15; B10: 8,9,10,11; C10: 8,9,10,11
    for r in range(7):
        col = 35 + r
        U[12:16, col] = wino_U[:, r]
        V[8:12, col] = wino_V[:, r]
        W[8:12, col] = wino_W[:, r]
        
    # Block 7: A10 B01 -> C11
    # A10: 8,9,10,11; B01: 4,5,6,7; C11: 12,13,14,15
    for r in range(7):
        col = 42 + r
        U[8:12, col] = wino_U[:, r]
        V[4:8, col] = wino_V[:, r]
        W[12:16, col] = wino_W[:, r]
        
    # Block 8: A11 B11 -> C11
    # A11: 12,13,14,15; B11: 12,13,14,15; C11: 12,13,14,15
    for r in range(7):
        col = 49 + r
        U[12:16, col] = wino_U[:, r]
        V[12:16, col] = wino_V[:, r]
        W[12:16, col] = wino_W[:, r]
        
    # Now the 5th column of B (index 16-19 in V, 16-19 in W)
    # C_col4 = A * B_col4
    # B_col4 is 4x1.
    # We split A into 4x2 and 4x2.
    # C_col4 = A[:, 0:2] B[0:2, 4] + A[:, 2:4] B[2:4, 4]
    # Each is a 4x2x1 multiplication, rank 8.
    
    # Block 9: A[:, 0:2] B[0:2, 4] -> C_col4
    # A[:, 0:2] is 4x2. B[0:2, 4] is 2x1.
    # C_col4 is 4x1.
    # This is 4 scalar multiplications? No, 8.
    # C_i = A_i0 B_0 + A_i1 B_1
    # This is 4 independent equations, each with 2 terms.
    # So 8 multiplications.
    
    # U for this part: 16x8
    # V for this part: 20x8 (only indices 16,17 are used)
    # W for this part: 20x8 (only indices 16,17,18,19 are used)
    
    for i in range(4):
        for j in range(2):
            col = 56 + i * 2 + j
            U[i * 2 + j, col] = 1 # A_i, j
            V[16 + j, col] = 1 # B_j, 4
            W[16 + i, col] = 1 # C_i, 4
            
    # Block 10: A[:, 2:4] B[2:4, 4] -> C_col4
    for i in range(4):
        for j in range(2):
            col = 60 + i * 2 + j
            U[i * 2 + 2 + j, col] = 1 # A_i, 2+j
            V[18 + j, col] = 1 # B_2+j, 4
            W[16 + i, col] = 1 # C_i, 4
            
    return {"U": U, "V": V, "W": W}
