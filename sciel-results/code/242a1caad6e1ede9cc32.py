import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen decomposition for 2x2x2
    # Returns lists of (idx_a, coeff), (idx_b, coeff), (idx_c, coeff)
    # idx_a is in [0, 4) for 2x2 matrix
    # idx_b is in [0, 4) for 2x2 matrix
    # idx_c is in [0, 4) for 2x2 matrix
    
    def strassen_2x2():
        terms = []
        # M1 = (a+d)(e+h)
        terms.append(([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]))
        # M2 = (c+d)e
        terms.append(([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]))
        # M3 = a(f-h)
        terms.append(([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]))
        # M4 = d(g-e)
        terms.append(([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]))
        # M5 = (a+b)h
        terms.append(([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]))
        # M6 = (c-a)(e+f)
        terms.append(([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]))
        # M7 = (b-d)(g+h)
        terms.append(([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1), (3,1)]))
        return terms

    strassen_terms = strassen_2x2()
    
    # 4x4x4 Strassen
    # We have 7 block products S1..S7
    # Each Sk is a 2x2 matrix product
    # We decompose each Sk into 7 scalar products
    # Total 49 terms for first half, 49 for second half
    
    # Mapping of 2x2 block indices to 4x4 indices
    # A blocks: 00, 01, 10, 11
    # A_00 is A[0:2, 0:2], A_01 is A[0:2, 2:4], etc.
    # Index in 4x4: i*4 + j
    # Block (bi, bj) covers i in [2*bi, 2*bi+2), j in [2*bj, 2*bj+2)
    
    # For each of the 7 Strassen steps (k in 0..6):
    # We have linear combinations of A blocks and B blocks
    # Then we multiply them
    # Then we add the result to C blocks
    
    # Let's define the 7 steps explicitly
    # Step 1: (A00 + A11) * (B00 + B11)
    # Step 2: (A10 + A11) * B00
    # Step 3: A00 * (B01 - B11)
    # Step 4: A11 * (B10 - B00)
    # Step 5: (A00 + A01) * B11
    # Step 6: (A10 - A00) * (B00 + B01)
    # Step 7: (A01 - A11) * (B10 + B11)
    
    steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: list of (block_idx, coeff)
        # B_coeffs: list of (block_idx, coeff)
        # C_coeffs: list of (block_idx, coeff)
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]), # S1 -> C00, C11
        ([ (2,1), (3,1) ], [ (0,1) ], [ (2,1), (3,-1) ]),       # S2 -> C10, C11
        ([ (0,1) ], [ (1,1), (3,-1) ], [ (1,1), (3,1) ]),       # S3 -> C01, C11
        ([ (3,1) ], [ (2,1), (0,-1) ], [ (0,1), (2,1) ]),       # S4 -> C00, C10
        ([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1), (1,1) ]),       # S5 -> C00, C01
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ], [ (3,1) ]),       # S6 -> C11
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1), (3,1) ]) # S7 -> C00, C11
    ]
    
    # Note: The C_coeffs in the steps above are for the 2x2 block result.
    # We need to map these to the final 4x8 C matrix.
    # C blocks: 00, 01, 10, 11
    # C_00 is C[0:2, 0:2] (cols 0,1)
    # C_01 is C[0:2, 2:4] (cols 2,3) -> This is the second half for the first 2 rows
    # C_10 is C[2:4, 0:2] (cols 0,1) -> This is the second half for the last 2 rows?
    # No, C is 4x8.
    # C[0:2, 0:4] corresponds to B1 (cols 0-3 of B)
    # C[0:2, 4:8] corresponds to B2 (cols 4-7 of B)
    
    # So for the first half (B1):
    # C_00 (block 0,0 of C) -> C[0:2, 0:2]
    # C_01 (block 0,1 of C) -> C[0:2, 2:4]
    # C_10 (block 1,0 of C) -> C[2:4, 0:2]
    # C_11 (block 1,1 of C) -> C[2:4, 2:4]
    
    # For the second half (B2):
    # C'_00 -> C[0:2, 4:6]
    # C'_01 -> C[0:2, 6:8]
    # C'_10 -> C[2:4, 4:6]
    # C'_11 -> C[2:4, 6:8]
    
    col_offset = 0
    for half in range(2):
        for step_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
            # Decompose the 2x2 product into 7 scalar products
            for term_idx, (u_terms, v_terms, w_terms) in enumerate(strassen_terms):
                # u_terms: list of (local_idx, coeff) for A 2x2
                # v_terms: list of (local_idx, coeff) for B 2x2
                # w_terms: list of (local_idx, coeff) for C 2x2
                
                # We need to combine the block coefficients with the local coefficients
                # For U:
                for (bi, coeff_a) in a_coeffs:
                    for (li, coeff_l) in u_terms:
                        # bi is block index in 4x4 (0..3)
                        # li is local index in 2x2 (0..3)
                        # Map to global index in 4x4
                        # Block bi covers rows 2*(bi//2) to 2*(bi//2)+1, cols 2*(bi%2) to 2*(bi%2)+1
                        # Local index li is row li//2, col li%2
                        # Global row = 2*(bi//2) + li//2
                        # Global col = 2*(bi%2) + li%2
                        gi = 2*(bi//2) + li//2
                        gj = 2*(bi%2) + li%2
                        global_idx = gi*4 + gj
                        U[global_idx, col_offset] += coeff_a * coeff_l
                        
                # For V:
                for (bi, coeff_b) in b_coeffs:
                    for (li, coeff_l) in v_terms:
                        gi = 2*(bi//2) + li//2
                        gj = 2*(bi%2) + li%2
                        # B is 4x8.
                        # Block bi is in the first 4 columns (B1) or last 4 columns (B2)
                        # If half == 0, cols are 0..3
                        # If half == 1, cols are 4..7
                        global_row = gi
                        global_col = 4*half + gj
                        global_idx = global_row*8 + global_col
                        V[global_idx, col_offset] += coeff_b * coeff_l
                        
                # For W:
                for (bi, coeff_c) in c_coeffs:
                    for (li, coeff_l) in w_terms:
                        gi = 2*(bi//2) + li//2
                        gj = 2*(bi%2) + li%2
                        # C is 4x8.
                        # Block bi is in the first 4 columns (C1) or last 4 columns (C2)
                        global_row = gj # C[k, i] -> k is col, i is row
                        # Wait, W[k*n+i, r]. k is col index of C, i is row index of C.
                        # C block bi:
                        # bi=0 (C00): rows 0,1; cols 0,1 (in C1)
                        # bi=1 (C01): rows 0,1; cols 2,3 (in C1)
                        # bi=2 (C10): rows 2,3; cols 0,1 (in C1)
                        # bi=3 (C11): rows 2,3; cols 2,3 (in C1)
                        
                        # So for half 0:
                        # col = gj
                        # row = gi
                        # For half 1:
                        # col = 4 + gj
                        # row = gi
                        
                        global_k = 4*half + gj
                        global_i = gi
                        global_idx = global_k*4 + global_i
                        W[global_idx, col_offset] += coeff_c * coeff_l
                        
                col_offset += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
