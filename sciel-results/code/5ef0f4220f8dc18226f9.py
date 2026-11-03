import numpy as np

def construct():
    n, m, p = 4, 4, 9
    # We will build the decomposition by combining smaller decompositions.
    # 1. M_444: Rank 49 (Strassen)
    # 2. M_444: Rank 49 (Strassen)
    # 3. M_441: Rank 8 (Block Schoolbook)
    
    # Total Rank = 49 + 49 + 8 = 106
    
    # Helper to create Strassen M_222 (Rank 7)
    # U_222: (4, 7), V_222: (4, 7), W_222: (4, 7)
    # Indices for U: i*m+j -> 0..3
    # Indices for V: j*p+k -> 0..3
    # Indices for W: k*n+i -> 0..3
    
    def get_strassen_222():
        # Strassen multipliers
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Col 0: M1
        # A parts: A11(0), A22(3)
        # B parts: B11(0), B22(3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C parts: C11(0), C12(1), C21(2), C22(3)
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
        
        # Col 1: M2
        # A parts: A21(2), A22(3)
        # B parts: B11(0)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C parts: C21 += M2, C22 += M2
        W[2, 1] = 1; W[3, 1] = 1
        
        # Col 2: M3
        # A parts: A11(0)
        # B parts: B12(1), B22(3)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C parts: C12 += M3, C22 -= M3
        W[1, 2] = 1; W[3, 2] = -1
        
        # Col 3: M4
        # A parts: A22(3)
        # B parts: B21(2), B11(0)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C parts: C21 += M4, C11 -= M4
        W[2, 3] = 1; W[0, 3] = -1
        
        # Col 4: M5
        # A parts: A11(0), A12(1)
        # B parts: B22(3)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C parts: C12 += M5, C22 += M5
        W[1, 4] = 1; W[3, 4] = 1
        
        # Col 5: M6
        # A parts: A21(2), A11(0)
        # B parts: B11(0), B12(1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C parts: C21 += M6, C11 -= M6
        W[2, 5] = 1; W[0, 5] = -1
        
        # Col 6: M7
        # A parts: A12(1), A22(3)
        # B parts: B21(2), B22(3)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C parts: C12 += M7, C22 -= M7
        W[1, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    def get_schoolbook_221():
        # M_221: A(2x2), B(2x1), C(2x1)
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Col 0: A11 B1 -> C1
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Col 1: A12 B2 -> C1
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Col 2: A21 B1 -> C2
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Col 3: A22 B2 -> C2
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Build M_444 (Rank 49)
    # A(4x4), B(4x4), C(4x4)
    # Strassen uses 7 M_222.
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    U_s, V_s, W_s = get_strassen_222()
    
    # Map indices for 4x4 blocks
    # A indices: 0..15 (i*4+j)
    # B indices: 0..15 (j*4+k)
    # C indices: 0..15 (k*4+i)
    
    # For each of the 7 Strassen steps, we place a 2x2x2 block.
    # We need to define which 2x2 sub-blocks of A, B, C are involved.
    
    # Strassen Steps:
    # 1. (A11+A22)(B11+B22) -> C11, C12, C21, C22
    # 2. (A21+A22)B11 -> C21, C22
    # 3. A11(B12-B22) -> C12, C22
    # 4. A22(B21-B11) -> C21, C11
    # 5. (A11+A12)B22 -> C12, C22
    # 6. (A21-A11)(B11+B12) -> C21, C11
    # 7. (A12-A22)(B21+B22) -> C12, C22
    
    # We will iterate through the 7 steps.
    # For each step, we have a linear combination of A blocks and B blocks.
    # This corresponds to a specific pattern in U and V.
    # And a specific pattern in W.
    
    # Let's define the steps explicitly.
    # Step 1:
    # U: A11(0:4), A22(12:16) with coeff 1
    # V: B11(0:4), B22(12:16) with coeff 1
    # W: C11(0:4) +1, C12(4:8) -1, C21(8:12) -1, C22(12:16) +1
    
    steps = [
        # (A_indices, A_coeffs, B_indices, B_coeffs, C_indices, C_coeffs)
        # Indices are start, end in the 16-length vector.
        # A11: 0-4, A12: 4-8, A21: 8-12, A22: 12-16
        # B11: 0-4, B12: 4-8, B21: 8-12, B22: 12-16
        # C11: 0-4, C12: 4-8, C21: 8-12, C22: 12-16
        
        ([0, 12], [1, 1], [0, 12], [1, 1], [0, 4, 8, 12], [1, -1, -1, 1]),
        ([8, 12], [1, 1], [0], [1], [8, 12], [1, 1]),
        ([0], [1], [4, 12], [1, -1], [4, 12], [1, -1]),
        ([12], [1], [8, 0], [1, -1], [8, 0], [1, -1]),
        ([0, 4], [1, 1], [12], [1], [4, 12], [1, 1]),
        ([8, 0], [1, -1], [0, 4], [1, 1], [8, 0], [1, -1]),
        ([4, 12], [1, -1], [8, 12], [1, 1], [4, 12], [1, -1]),
    ]
    
    for step_idx, (a_inds, a_coeffs, b_inds, b_coeffs, c_inds, c_coeffs) in enumerate(steps):
        base_col = step_idx * 7
        for k in range(7):
            col = base_col + k
            # U
            for a_i, a_c in zip(a_inds, a_coeffs):
                # a_i is start index of a 4-block
                # We need to map the 2x2 Strassen U to the 4x4 U
                # U_s has shape (4, 7).
                # The 4 rows of U_s correspond to the 4 elements of the 2x2 A block.
                # We need to add these to the corresponding rows in U_444.
                for row_s in range(4):
                    if U_s[row_s, k] != 0:
                        # Map row_s to global row
                        # The 2x2 block starts at a_i.
                        # row_s 0->0, 1->1, 2->2, 3->3 relative to block.
                        global_row = a_i + row_s
                        U_444[global_row, col] += U_s[row_s, k] * a_c
                        
            # V
            for b_i, b_c in zip(b_inds, b_coeffs):
                for row_s in range(4):
                    if V_s[row_s, k] != 0:
                        global_row = b_i + row_s
                        V_444[global_row, col] += V_s[row_s, k] * b_c
                        
            # W
            for c_i, c_c in zip(c_inds, c_coeffs):
                for row_s in range(4):
                    if W_s[row_s, k] != 0:
                        global_row = c_i + row_s
                        W_444[global_row, col] += W_s[row_s, k] * c_c
                        
    # Build M_441 (Rank 8)
    # A(4x4), B(4x1), C(4x1)
    # Use 4 M_221 blocks.
    # A11 B1 -> C1, A12 B2 -> C1, A21 B1 -> C2, A22 B2 -> C2
    # B1 is rows 0-1 of B, B2 is rows 2-3 of B.
    # C1 is rows 0-1 of C, C2 is rows 2-3 of C.
    
    U_441 = np.zeros((16, 8))
    V_441 = np.zeros((4, 8))
    W_441 = np.zeros((4, 8))
    
    U_sb, V_sb, W_sb = get_schoolbook_221()
    
    # Block 1: A11(0-4), B1(0-2), C1(0-2)
    # Block 2: A12(4-8), B2(2-4), C1(0-2)
    # Block 3: A21(8-12), B1(0-2), C2(2-4)
    # Block 4: A22(12-16), B2(2-4), C2(2-4)
    
    blocks = [
        (0, 0, 0),
        (4, 2, 0),
        (8, 0, 2),
        (12, 2, 2)
    ]
    
    for block_idx, (a_start, b_start, c_start) in enumerate(blocks):
        base_col = block_idx * 4
        for k in range(4):
            col = base_col + k
            for row_s in range(4):
                if U_sb[row_s, k] != 0:
                    U_441[a_start + row_s, col] += U_sb[row_s, k]
            for row_s in range(2):
                if V_sb[row_s, k] != 0:
                    V_441[b_start + row_s, col] += V_sb[row_s, k]
            for row_s in range(2):
                if W_sb[row_s, k] != 0:
                    W_441[c_start + row_s, col] += W_sb[row_s, k]
                    
    # Combine into M_449
    # B is 4x9.
    # B1 (cols 0-3) -> M_444
    # B2 (cols 4-7) -> M_444
    # B3 (col 8) -> M_441
    
    U_total = np.zeros((16, 106))
    V_total = np.zeros((36, 106))
    W_total = np.zeros((36, 106))
    
    # Part 1: B1 (cols 0-3)
    # U is same
    U_total[:, 0:49] = U_444
    # V: B1 is rows 0-3 of B. In V_444, rows are j*4+k.
    # For B1, k is 0-3. So rows 0-15 of V_444 map to rows 0-15 of V_total?
    # Wait, V_444 has shape (16, 49).
    # V_total has shape (36, 106).
    # B1 corresponds to k=0..3.
    # In V_444, index is j*4+k.
    # In V_total, index is j*9+k.
    # So for j=0..3, k=0..3:
    # V_total[j*9+k, col] = V_444[j*4+k, col]
    for j in range(4):
        for k in range(4):
            V_total[j*9+k, 0:49] = V_444[j*4+k, 0:49]
            
    # W: C1 is cols 0-3.
    # W_444 shape (16, 49). Index k*4+i.
    # W_total shape (36, 106). Index k*4+i? No, W is p*n.
    # W_total index is k*n+i. n=4.
    # So for k=0..3, i=0..3:
    # W_total[k*4+i, col] = W_444[k*4+i, col]
    W_total[0:16, 0:49] = W_444[0:16, 0:49]
    
    # Part 2: B2 (cols 4-7)
    U_total[:, 49:98] = U_444
    for j in range(4):
        for k in range(4):
            # B2 col k corresponds to global col k+4
            V_total[j*9+(k+4), 49:98] = V_444[j*4+k, 0:49]
            
    # C2 cols 4-7
    for k in range(4):
        # C2 col k corresponds to global col k+4
        W_total[(k+4)*4: (k+4)*4+4, 49:98] = W_444[k*4: k*4+4, 0:49]
        
    # Part 3: B3 (col 8)
    U_total[:, 98:106] = U_441
    for j in range(4):
        # B3 col 8
        V_total[j*9+8, 98:106] = V_441[j, 0:8]
        
    # C3 col 8
    for k in range(4):
        W_total[k*4: k*4+4, 98:106] = W_441[k, 0:8]
        
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
