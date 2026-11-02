import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We decompose the multiplication into blocks.
    # A is 3x4, B is 4x8.
    # Split A into A_top (2x4) and A_bottom (1x4).
    # Split B into B_left (4x4) and B_right (4x4).
    # C_top_left (2x4) = A_top @ B_left
    # C_top_right (2x4) = A_top @ B_right
    # C_bottom_left (1x4) = A_bottom @ B_left
    # C_bottom_right (1x4) = A_bottom @ B_right
    
    # Rank for 2x4x4 using Strassen on 2x2 blocks:
    # 2x4x4 can be seen as 4 multiplications of 2x2x2.
    # Rank of 2x2x2 is 7.
    # Total rank = 4 * 7 = 28.
    
    # Rank for 1x4x4 using Schoolbook:
    # 1x4x4 is 4 dot products of length 4.
    # Rank = 4 * 4 = 16.
    
    # Total Rank = 28 + 28 + 16 + 16 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)] # 12 x 88
    V = [[0] * rank for _ in range(m * p)] # 32 x 88
    W = [[0] * rank for _ in range(p * n)] # 24 x 88
    
    # We need to map the indices correctly.
    # U index: i*m + j, where i in [0,2], j in [0,3]
    # V index: j*p + k, where j in [0,3], k in [0,7]
    # W index: k*n + i, where k in [0,7], i in [0,2]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_indices: list of 4 (i,j) pairs for A (2x2)
    # b_indices: list of 4 (j,k) pairs for B (2x2)
    # c_indices: list of 4 (k,i) pairs for C (2x2)
    def add_strassen_2x2(a_idx, b_idx, c_idx, start_col):
        nonlocal col
        # Strassen 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # A indices: [ (i1,j1), (i1,j2), (i2,j1), (i2,j2) ]
        # B indices: [ (j1,k1), (j1,k2), (j2,k1), (j2,k2) ]
        # C indices: [ (k1,i1), (k1,i2), (k2,i1), (k2,i2) ]
        
        # Note: The standard Strassen formula computes C = AB.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add 7 terms to U, V, W.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # U: A11, A22 -> +1
        # V: B11, B22 -> +1
        # W: C11, C22 -> +1 (from M1), C22 also from M1?
        # Wait, M1 contributes to C11 and C22.
        # C11 += M1, C22 += M1.
        # So W[C11] += 1, W[C22] += 1.
        
        # Let's define the contributions for each of the 7 multiplications.
        
        # 1. M1 = (A11+A22)(B11+B22)
        # A: A11(+1), A22(+1)
        # B: B11(+1), B22(+1)
        # C: C11(+1), C22(+1)
        
        # 2. M2 = (A21+A22)B11
        # A: A21(+1), A22(+1)
        # B: B11(+1)
        # C: C21(+1), C22(-1)
        
        # 3. M3 = A11(B21-B22)
        # A: A11(+1)
        # B: B21(+1), B22(-1)
        # C: C12(+1), C21(+1)
        
        # 4. M4 = A22(B12-B11)
        # A: A22(+1)
        # B: B12(+1), B11(-1)
        # C: C11(+1), C21(+1)
        
        # 5. M5 = (A11+A12)B22
        # A: A11(+1), A12(+1)
        # B: B22(+1)
        # C: C11(-1), C12(+1)
        
        # 6. M6 = (A21-A11)(B11+B12)
        # A: A21(+1), A11(-1)
        # B: B11(+1), B12(+1)
        # C: C22(+1)
        
        # 7. M7 = (A12-A22)(B21+B22)
        # A: A12(+1), A22(-1)
        # B: B21(+1), B22(+1)
        # C: C11(+1), C22(-1)
        
        # We will add these 7 terms.
        
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            # a_coeffs: list of (index, coeff)
            # b_coeffs: list of (index, coeff)
            # c_coeffs: list of (index, coeff)
            
            for idx, coeff in a_coeffs:
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in b_coeffs:
                if coeff != 0:
                    V[idx][col] += coeff
            for idx, coeff in c_coeffs:
                if coeff != 0:
                    W[idx][col] += coeff
            col += 1
        
        # Map local 2x2 indices to global indices
        # a_idx: [ (i1,j1), (i1,j2), (i2,j1), (i2,j2) ]
        # Global A index: i*m + j
        # b_idx: [ (j1,k1), (j1,k2), (j2,k1), (j2,k2) ]
        # Global B index: j*p + k
        # c_idx: [ (k1,i1), (k1,i2), (k2,i1), (k2,i2) ]
        # Global C index: k*n + i
        
        A00 = a_idx[0]; A01 = a_idx[1]; A10 = a_idx[2]; A11 = a_idx[3]
        B00 = b_idx[0]; B01 = b_idx[1]; B10 = b_idx[2]; B11 = b_idx[3]
        C00 = c_idx[0]; C01 = c_idx[1]; C10 = c_idx[2]; C11 = c_idx[3]
        
        # 1. M1
        add_term([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(C00, 1), (C11, 1)])
        
        # 2. M2
        add_term([(A10, 1), (A11, 1)], [(B00, 1)], [(C10, 1), (C11, -1)])
        
        # 3. M3
        add_term([(A00, 1)], [(B10, 1), (B11, -1)], [(C01, 1), (C10, 1)])
        
        # 4. M4
        add_term([(A11, 1)], [(B01, 1), (B00, -1)], [(C00, 1), (C10, 1)])
        
        # 5. M5
        add_term([(A00, 1), (A01, 1)], [(B11, 1)], [(C00, -1), (C01, 1)])
        
        # 6. M6
        add_term([(A10, 1), (A00, -1)], [(B00, 1), (B01, 1)], [(C11, 1)])
        
        # 7. M7
        add_term([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)], [(C00, 1), (C11, -1)])
        
        return col

    # Helper to add a 1x4x4 Schoolbook block
    def add_schoolbook_1x4(a_idx, b_idx, c_idx, start_col):
        # a_idx: list of 4 j indices (row 0 of A)
        # b_idx: list of 4 (j,k) pairs
        # c_idx: list of 4 k indices (row 0 of C)
        
        # C_k = sum_j A_j B_jk
        # We add 16 terms.
        
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for idx, coeff in a_coeffs:
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in b_coeffs:
                if coeff != 0:
                    V[idx][col] += coeff
            for idx, coeff in c_coeffs:
                if coeff != 0:
                    W[idx][col] += coeff
            col += 1

        # a_idx is actually just the j indices for the single row i=2 (0-indexed)
        # But we need the full global index for U.
        # Let's assume a_idx passed is the list of global U indices for the 4 elements.
        # b_idx passed is list of global V indices.
        # c_idx passed is list of global W indices.
        
        # For each k (0..3), C_k = sum_j A_j B_jk
        for k in range(4):
            # Term for A_j B_jk contributing to C_k
            # Actually, we can just add 4 terms for each k?
            # No, the tensor is sum_{j,k} e_j (A) \otimes e_{jk} (B) \otimes e_k (C)
            # So for each j, k, we have a term A_j * B_{jk} * C_k.
            
            for j in range(4):
                # Global U index for A_j
                u_idx = a_idx[j]
                # Global V index for B_jk
                v_idx = b_idx[j*4 + k] # Assuming b_idx is flattened row-major
                # Global W index for C_k
                w_idx = c_idx[k]
                
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                
        return col

    # Block 1: A_top (rows 0,1) x B_left (cols 0-3)
    # A indices: i in [0,1], j in [0,3]
    # B indices: j in [0,3], k in [0,3]
    # C indices: k in [0,3], i in [0,1]
    
    # We split A_top into two 2x2 blocks? No, A_top is 2x4.
    # We view A_top as [A_top_left (2x2), A_top_right (2x2)]
    # B_left as [B_top_left (2x2), B_bottom_left (2x2)] in rows?
    # No, B is 4x4.
    # B_left = [[B00, B01], [B10, B11]] where each is 2x2.
    
    # We need to perform 2x4x4 multiplication.
    # This is 4 2x2x2 multiplications.
    # 1. A_top_left * B_top_left -> C_top_left (2x2)
    # 2. A_top_left * B_bottom_left -> C_bottom_left (2x2)
    # 3. A_top_right * B_top_left -> C_top_right (2x2)
    # 4. A_top_right * B_bottom_left -> C_bottom_right (2x2)
    
    # Indices for A_top (rows 0,1 of A)
    # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
    # A02: (0,2), A03: (0,3), A12: (1,2), A13: (1,3)
    
    # Indices for B_left (cols 0-3 of B)
    # B00: (0,0), B01: (0,1), B10: (1,0), B11: (1,1)
    # B02: (0,2), B03: (0,3), B12: (1,2), B13: (1,3)
    # B20: (2,0), B21: (2,1), B30: (3,0), B31: (3,1)
    # B22: (2,2), B23: (2,3), B32: (3,2), B33: (3,3)
    
    # Wait, B is 4x4.
    # B_left is the first 4 columns.
    # Rows 0-3 of B.
    
    # Let's define the 4 2x2x2 blocks for the 2x4x4 multiplication.
    
    # Block 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    a_idx_11 = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
    b_idx_11 = [0*p+0, 0*p+1, 1*p+0, 1*p+1]
    c_idx_11 = [0*n+0, 0*n+1, 1*n+0, 1*n+1]
    add_strassen_2x2(a_idx_11, b_idx_11, c_idx_11, col)
    
    # Block 1.2: A[0:2, 0:2] * B[2:4, 0:2] -> C[2:4, 0:2]
    a_idx_12 = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
    b_idx_12 = [2*p+0, 2*p+1, 3*p+0, 3*p+1]
    c_idx_12 = [2*n+0, 2*n+1, 3*n+0, 3*n+1]
    add_strassen_2x2(a_idx_12, b_idx_12, c_idx_12, col)
    
    # Block 1.3: A[0:2, 2:4] * B[0:2, 0:2] -> C[0:2, 2:4]
    a_idx_13 = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
    b_idx_13 = [0*p+0, 0*p+1, 1*p+0, 1*p+1]
    c_idx_13 = [0*n+2, 0*n+3, 1*n+2, 1*n+3]
    add_strassen_2x2(a_idx_13, b_idx_13, c_idx_13, col)
    
    # Block 1.4: A[0:2, 2:4] * B[2:4, 0:2] -> C[2:4, 2:4]
    a_idx_14 = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
    b_idx_14 = [2*p+0, 2*p+1, 3*p+0, 3*p+1]
    c_idx_14 = [2*n+2, 2*n+3, 3*n+2, 3*n+3]
    add_strassen_2x2(a_idx_14, b_idx_14, c_idx_14, col)
    
    # Now B_right (cols 4-7)
    # Block 2.1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    a_idx_21 = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
    b_idx_21 = [0*p+4, 0*p+5, 1*p+4, 1*p+5]
    c_idx_21 = [0*n+4, 0*n+5, 1*n+4, 1*n+5]
    add_strassen_2x2(a_idx_21, b_idx_21, c_idx_21, col)
    
    # Block 2.2: A[0:2, 0:2] * B[2:4, 4:6] -> C[2:4, 4:6]
    a_idx_22 = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
    b_idx_22 = [2*p+4, 2*p+5, 3*p+4, 3*p+5]
    c_idx_22 = [2*n+4, 2*n+5, 3*n+4, 3*n+5]
    add_strassen_2x2(a_idx_22, b_idx_22, c_idx_22, col)
    
    # Block 2.3: A[0:2, 2:4] * B[0:2, 4:6] -> C[0:2, 6:8]
    a_idx_23 = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
    b_idx_23 = [0*p+4, 0*p+5, 1*p+4, 1*p+5]
    c_idx_23 = [0*n+6, 0*n+7, 1*n+6, 1*n+7]
    add_strassen_2x2(a_idx_23, b_idx_23, c_idx_23, col)
    
    # Block 2.4: A[0:2, 2:4] * B[2:4, 4:6] -> C[2:4, 6:8]
    a_idx_24 = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
    b_idx_24 = [2*p+4, 2*p+5, 3*p+4, 3*p+5]
    c_idx_24 = [2*n+6, 2*n+7, 3*n+6, 3*n+7]
    add_strassen_2x2(a_idx_24, b_idx_24, c_idx_24, col)
    
    # Now A_bottom (row 2)
    # A[2, 0:4] * B[0:4, 0:4] -> C[0:4, 2]
    # This is 1x4x4.
    a_idx_31 = [2*m+0, 2*m+1, 2*m+2, 2*m+3]
    b_idx_31 = [0*p+0, 0*p+1, 0*p+2, 0*p+3, 1*p+0, 1*p+1, 1*p+2, 1*p+3, 2*p+0, 2*p+1, 2*p+2, 2*p+3, 3*p+0, 3*p+1, 3*p+2, 3*p+3]
    c_idx_31 = [0*n+2, 1*n+2, 2*n+2, 3*n+2]
    add_schoolbook_1x4(a_idx_31, b_idx_31, c_idx_31, col)
    
    # A[2, 0:4] * B[0:4, 4:8] -> C[4:8, 2]
    a_idx_32 = [2*m+0, 2*m+1, 2*m+2, 2*m+3]
    b_idx_32 = [0*p+4, 0*p+5, 0*p+6, 0*p+7, 1*p+4, 1*p+5, 1*p+6, 1*p+7, 2*p+4, 2*p+5, 2*p+6, 2*p+7, 3*p+4, 3*p+5, 3*p+6, 3*p+7]
    c_idx_32 = [4*n+2, 5*n+2, 6*n+2, 7*n+2]
    add_schoolbook_1x4(a_idx_32, b_idx_32, c_idx_32, col)
    
    return {"U": U, "V": V, "W": W}
