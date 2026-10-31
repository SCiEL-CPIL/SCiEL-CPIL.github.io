import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to map block (br, bc) and local (lr, lc) to global (i, j)
    def block_to_global(br, bc, lr, lc):
        return 2*br + lr, 2*bc + lc

    # Define the 7 X_q and Y_q matrices for the outer Strassen step
    # Each is a 2x2 matrix of 4x4 coefficient matrices
    # X_q[i][j] is a 4x4 matrix of coeffs for A
    # We define them as lists of tuples of coeffs for A11, A12, A21, A22
    # Block indices: 0=11, 1=12, 2=21, 3=22
    
    # Coefficients for X_q (blocks)
    # q=0: A11+A22 -> (1,0,0,1) for all 4 entries? No, X_q is 2x2 matrix of blocks.
    # X_q[0,0] is A11 part, X_q[0,1] is A12 part, etc.
    
    # Let's define the block coefficients for X_q and Y_q
    # X_q is 2x2. X_q[r][c] is a linear combo of A blocks.
    # Actually, X_q itself is a 2x2 matrix.
    # X_q = [[A11, A12], [A21, A22]] transformed.
    
    # Outer Strassen X definitions (coeffs for A11, A12, A21, A22)
    # X_q is a 2x2 matrix.
    # X_0 = [[A11, 0], [0, A22]] + [[0,0],[0,0]]? No.
    # X_0 = A11 + A22 means the matrix X_0 is [[A11, 0], [0, A22]]? No.
    # A11 is top-left block. A22 is bottom-right.
    # So X_0 = [[A11, 0], [0, A22]]? No, A11+A22 is not a block diagonal.
    # Wait, A is [[A11, A12], [A21, A22]].
    # X_0 = A11 + A22. This is a 2x2 matrix where the top-left block is A11 and bottom-right is A22?
    # No, A11 and A22 are 2x2 matrices.
    # X_0 = A11 + A22 is a 2x2 matrix.
    # The entries of X_0 are:
    # X_0[0,0] = A11[0,0] + A22[0,0]
    # X_0[0,1] = A11[0,1] + A22[0,1]
    # X_0[1,0] = A11[1,0] + A22[1,0]
    # X_0[1,1] = A11[1,1] + A22[1,1]
    
    # So, to get the 4x4 coefficient matrix for A, we sum the coefficient matrices of the blocks involved.
    # Let's define block coefficient matrices.
    # B_00 (A11) is 4x4 with 1s at (0,0),(0,1),(1,0),(1,1)
    # B_01 (A12) is 4x4 with 1s at (0,2),(0,3),(1,2),(1,3)
    # B_10 (A21) is 4x4 with 1s at (2,0),(2,1),(3,0),(3,1)
    # B_11 (A22) is 4x4 with 1s at (2,2),(2,3),(3,2),(3,3)
    
    def get_block_mat(br, bc):
        mat = np.zeros((4,4))
        for r in range(2):
            for c in range(2):
                i = 2*br + r
                j = 2*bc + c
                mat[i,j] = 1
        return mat

    B00 = get_block_mat(0,0)
    B01 = get_block_mat(0,1)
    B10 = get_block_mat(1,0)
    B11 = get_block_mat(1,1)

    # Define X_q as 2x2 array of 4x4 matrices
    # X_q[r][c] is the coefficient matrix for the entry at (r,c) of the 2x2 matrix X_q
    # X_q is formed by linear combos of blocks.
    # X_0 = A11 + A22.
    # X_0[0,0] = A11[0,0] + A22[0,0] -> Coeffs: B00[0,0] + B11[0,0]
    # Actually, it's simpler to say:
    # X_0 is the sum of matrices A11 and A22.
    # So X_0[0,0] = B00[0,0] + B11[0,0]
    # X_0[0,1] = B00[0,1] + B11[0,1]
    # ...
    
    # Let's define the block coefficients for each of the 7 X and Y
    # X_q is a 2x2 matrix.
    # We define it by specifying which blocks are added with what sign.
    # e.g. X_0: A11 (+1), A22 (+1).
    # X_1: A21 (+1), A22 (+1).
    # X_2: A11 (+1), A12 (-1).
    # X_3: A22 (+1).
    # X_4: A11 (+1), A21 (+1).
    # X_5: A12 (+1), A22 (+1).
    # X_6: A11 (+1), A21 (-1).
    
    X_blocks = [
        [(1,0), (1,0)], # X_0: A11+A22. Top-left is A11, Bot-right is A22.
        # Wait, A11 is top-left block. A22 is bottom-right block.
        # X_0[0,0] comes from A11[0,0] and A22[0,0].
        # So for X_0, the block at (0,0) is A11, block at (1,1) is A22.
        # So X_0[0,0] coeff is B00[0,0]+B11[0,0].
        # This is getting confusing.
        # Let's just build the 4x4 matrices directly.
    ]
    
    # Let's define X_q and Y_q as 4x4 matrices of coefficients.
    # X_q is a 2x2 matrix of 2x2 blocks.
    # So X_q is a 4x4 matrix of coefficients.
    
    def build_X(q):
        X = np.zeros((4,4))
        if q == 0: # A11 + A22
            X += B00 + B11
        elif q == 1: # A21 + A22
            X += B10 + B11
        elif q == 2: # A11 - A12
            X += B00 - B01
        elif q == 3: # A22
            X += B11
        elif q == 4: # A11 + A21
            X += B00 + B10
        elif q == 5: # A12 + A22
            X += B01 + B11
        elif q == 6: # A11 - A21
            X += B00 - B10
        return X

    def build_Y(q):
        Y = np.zeros((4,4))
        if q == 0: # B11 + B22
            Y += B00 + B11
        elif q == 1: # B11
            Y += B00
        elif q == 2: # B12 - B22
            Y += B01 - B11
        elif q == 3: # B21 + B22
            Y += B10 + B11
        elif q == 4: # B11 - B22
            Y += B00 - B11
        elif q == 5: # B22 - B21
            Y += B11 - B10
        elif q == 6: # B11 + B12
            Y += B00 + B01
        return Y

    # Inner Strassen combinations
    # For a given X (4x4) and Y (4x4), we compute 7 products.
    # X = [[x00, x01], [x10, x11]] where xij are 2x2 blocks.
    # But we can just treat X as a 4x4 matrix and extract the 2x2 blocks.
    # x00 = X[0:2, 0:2]
    # x01 = X[0:2, 2:4]
    # x10 = X[2:4, 0:2]
    # x11 = X[2:4, 2:4]
    
    # Inner Strassen formulas for X, Y:
    # P1: (x00+x11) * (y00+y11)
    # P2: (x10+x11) * y00
    # P3: (x00-x01) * (y01-y11)
    # P4: x11 * (y10+y11)
    # P5: (x00+x10) * (y00-y11)
    # P6: (x01+x11) * (y11-y10)
    # P7: (x00-x10) * (y00+y01)
    
    # We need to map these to U and V columns.
    
    # Outer Strassen combinations for M_q to C blocks
    # C11 = M0 + M3 - M4 - M6
    # C12 = M1 + M2
    # C21 = M5 + M6
    # C22 = M0 - M2 + M3 + M5
    
    # M_q is a 2x2 matrix of 2x2 blocks.
    # M_q[0,0] contributes to C[0:2, 0:2] (C11)
    # M_q[0,1] contributes to C[0:2, 2:4] (C12)
    # M_q[1,0] contributes to C[2:4, 0:2] (C21)
    # M_q[1,1] contributes to C[2:4, 2:4] (C22)
    
    # Inner Strassen output M_q entries:
    # M_q[0,0] = P1 + P4 - P5 - P7
    # M_q[0,1] = P3 + P5
    # M_q[1,0] = P2 + P7
    # M_q[1,1] = P1 - P3 + P4 + P6
    
    # We iterate over q (0..6) and s (0..6) for inner products.
    # r = 7*q + s
    
    # Precompute X and Y for each q
    Xs = [build_X(q) for q in range(7)]
    Ys = [build_Y(q) for q in range(7)]
    
    # Inner Strassen coefficient vectors for X and Y
    # s=0 (P1): X: x00+x11, Y: y00+y11
    # s=1 (P2): X: x10+x11, Y: y00
    # s=2 (P3): X: x00-x01, Y: y01-y11
    # s=3 (P4): X: x11, Y: y10+y11
    # s=4 (P5): X: x00+x10, Y: y00-y11
    # s=5 (P6): X: x01+x11, Y: y11-y10
    # s=6 (P7): X: x00-x10, Y: y00+y01
    
    # We define the linear combinations of the 4 blocks (x00, x01, x10, x11)
    # as 4-tuples of coefficients.
    # x00 is at [0:2, 0:2], etc.
    
    inner_X_coeffs = [
        (1, 0, 0, 1), # P1: x00 + x11
        (0, 0, 1, 1), # P2: x10 + x11
        (1, -1, 0, 0),# P3: x00 - x01
        (0, 0, 0, 1), # P4: x11
        (1, 0, 1, 0), # P5: x00 + x10
        (0, 1, 0, 1), # P6: x01 + x11
        (1, 0, -1, 0) # P7: x00 - x10
    ]
    
    inner_Y_coeffs = [
        (1, 0, 0, 1), # P1: y00 + y11
        (1, 0, 0, 0), # P2: y00
        (0, 1, 0, -1),# P3: y01 - y11
        (0, 0, 1, 1), # P4: y10 + y11
        (1, 0, 0, -1),# P5: y00 - y11
        (0, 0, -1, 1),# P6: y11 - y10
        (1, 1, 0, 0)  # P7: y00 + y01
    ]
    
    # Map inner product s to M_q entries
    # M_q[0,0] = P1 + P4 - P5 - P7
    # M_q[0,1] = P3 + P5
    # M_q[1,0] = P2 + P7
    # M_q[1,1] = P1 - P3 + P4 + P6
    
    # s=0 (P1): M00(+1), M11(+1)
    # s=1 (P2): M10(+1)
    # s=2 (P3): M01(+1), M11(-1)
    # s=3 (P4): M00(+1), M11(+1)
    # s=4 (P5): M00(-1), M01(+1)
    # s=5 (P6): M11(+1)
    # s=6 (P7): M00(-1), M10(+1)
    
    m_entry_coeffs = [
        (1, 0, 0, 1), # P1: M00, M11
        (0, 0, 1, 0), # P2: M10
        (0, 1, 0, -1),# P3: M01, M11
        (1, 0, 0, 1), # P4: M00, M11
        (-1, 1, 0, 0),# P5: M00, M01
        (0, 0, 0, 1), # P6: M11
        (-1, 0, 1, 0) # P7: M00, M10
    ]
    
    # Map M_q entries to C blocks
    # C11 = M0 + M3 - M4 - M6
    # C12 = M1 + M2
    # C21 = M5 + M6
    # C22 = M0 - M2 + M3 + M5
    
    # q=0 (M0): C11(+1), C22(+1)
    # q=1 (M1): C12(+1)
    # q=2 (M2): C12(+1), C22(-1)
    # q=3 (M3): C11(+1), C22(+1)
    # q=4 (M4): C11(-1)
    # q=5 (M5): C21(+1), C22(+1)
    # q=6 (M6): C11(-1), C21(+1)
    
    c_block_coeffs = [
        (1, 0, 0, 1), # M0: C11, C22
        (0, 1, 0, 0), # M1: C12
        (0, 1, 0, -1),# M2: C12, C22
        (1, 0, 0, 1), # M3: C11, C22
        (-1, 0, 0, 0),# M4: C11
        (0, 0, 1, 1), # M5: C21, C22
        (-1, 0, 1, 0) # M6: C11, C21
    ]
    
    # Block to global index mapping for C
    # C11: [0:2, 0:2]
    # C12: [0:2, 2:4]
    # C21: [2:4, 0:2]
    # C22: [2:4, 2:4]
    
    def get_c_block_mat(block_idx):
        mat = np.zeros((4,4))
        if block_idx == 0: # C11
            mat[0:2, 0:2] = 1
        elif block_idx == 1: # C12
            mat[0:2, 2:4] = 1
        elif block_idx == 2: # C21
            mat[2:4, 0:2] = 1
        elif block_idx == 3: # C22
            mat[2:4, 2:4] = 1
        return mat

    C_blocks = [get_c_block_mat(i) for i in range(4)]

    # Now construct U, V, W
    for q in range(7):
        X = Xs[q]
        Y = Ys[q]
        # Extract 2x2 blocks from X and Y
        x00 = X[0:2, 0:2]
        x01 = X[0:2, 2:4]
        x10 = X[2:4, 0:2]
        x11 = X[2:4, 2:4]
        
        y00 = Y[0:2, 0:2]
        y01 = Y[0:2, 2:4]
        y10 = Y[2:4, 0:2]
        y11 = Y[2:4, 2:4]
        
        x_blocks = [x00, x01, x10, x11]
        y_blocks = [y00, y01, y10, y11]
        
        for s in range(7):
            r = 7*q + s
            
            # Compute U_r
            U_r = np.zeros((4,4))
            coeffs = inner_X_coeffs[s]
            for i in range(4):
                U_r += coeffs[i] * x_blocks[i]
            
            # Compute V_r
            V_r = np.zeros((4,4))
            coeffs = inner_Y_coeffs[s]
            for i in range(4):
                V_r += coeffs[i] * y_blocks[i]
            
            # Compute W_r
            # First, determine contribution to M_q entries
            m_coeffs = m_entry_coeffs[s]
            # m_coeffs[0] for M00, m_coeffs[1] for M01, m_coeffs[2] for M10, m_coeffs[3] for M11
            # M00 is block (0,0) of M_q
            # M01 is block (0,1) of M_q
            # M10 is block (1,0) of M_q
            # M11 is block (1,1) of M_q
            
            # Now map M_q blocks to C blocks
            c_coeffs = c_block_coeffs[q]
            # c_coeffs[0] for C11, c_coeffs[1] for C12, c_coeffs[2] for C21, c_coeffs[3] for C22
            
            W_r = np.zeros((4,4))
            
            # M00 (block 0,0) contributes to C blocks based on c_coeffs
            # M00 is at local (0,0) in M_q.
            # M_q is a 2x2 block matrix.
            # M00 corresponds to the top-left 2x2 of M_q.
            # When M_q is placed in C, M00 goes to C11 (if q contributes to C11).
            # The position of M00 in C is determined by the C block.
            # If M_q contributes to C11, M00 is at C[0:2, 0:2].
            # If M_q contributes to C12, M00 is at C[0:2, 2:4].
            # If M_q contributes to C21, M00 is at C[2:4, 0:2].
            # If M_q contributes to C22, M00 is at C[2:4, 2:4].
            
            # So, for each M_q entry (mi, mj) with coeff mc:
            # For each C block (ci) with coeff cc:
            # The contribution is mc * cc * (identity on the 2x2 subblock?)
            # No, M00 is a 2x2 matrix. It adds to the corresponding 2x2 subblock of C.
            # So W_r should have mc * cc * 1 in the corresponding 2x2 subblock.
            
            m_entry_to_c_block = {
                0: 0, # M00 -> C11 (if q says so)
                1: 1, # M01 -> C12
                2: 2, # M10 -> C21
                3: 3  # M11 -> C22
            }
            
            for mi in range(4):
                if m_coeffs[mi] == 0: continue
                c_block_idx = m_entry_to_c_block[mi]
                # Check if q contributes to this C block
                cc = c_coeffs[c_block_idx]
                if cc == 0: continue
                
                # The M_q entry mi is a 2x2 block.
                # It maps to the 2x2 subblock of C_block.
                # The subblock position in C is fixed by c_block_idx.
                # The subblock position within M_q is (mi//2, mi%2).
                # Wait, M00 is (0,0), M01 is (0,1), M10 is (1,0), M11 is (1,1).
                # So the local row is mi//2, local col is mi%2.
                # The global row in C is 2*(c_block_idx//2) + (mi//2)
                # The global col in C is 2*(c_block_idx%2) + (mi%2)
                
                gr = 2*(c_block_idx//2) + (mi//2)
                gc = 2*(c_block_idx%2) + (mi%2)
                
                W_r[gr:gr+2, gc:gc+2] += m_coeffs[mi] * cc
            
            # Store U, V, W
            U[:, r] = U_r.flatten()
            V[:, r] = V_r.flatten()
            W[:, r] = W_r.flatten()
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
