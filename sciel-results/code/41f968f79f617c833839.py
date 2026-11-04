import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 110

    U_cols = []
    V_cols = []
    W_cols = []

    def add_block_strassen(X_coeffs, Y_coeffs, W_targets):
        """
        Add rank-7 decomposition of product of two 2x2 matrices X and Y.
        X is a linear combination of 2x2 blocks of A.
        Y is a linear combination of 2x2 blocks of B.
        W_targets is a list of (target_block_idx, coeff).
        target_block_idx corresponds to C blocks:
        0: C[0:2, 0:2]
        1: C[0:2, 2:4]
        2: C[2:4, 0:2]
        3: C[2:4, 2:4]
        """
        # Block definitions for A (4x4)
        # A00: (0,0)-(1,1), A01: (0,2)-(1,3), A10: (2,0)-(3,1), A11: (2,2)-(3,3)
        # Indices: 0, 1, 2, 3
        A_blocks = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)]
        ]
        
        # Block definitions for B (4x6)
        # We only use the first 4 columns for the 4x4 part.
        # B00: (0,0)-(1,1), B01: (0,2)-(1,3), B10: (2,0)-(3,1), B11: (2,2)-(3,3)
        B_blocks = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)]
        ]
        
        # C blocks for the 4x4 part (first 4 cols)
        # C00: (0,0)-(1,1), C01: (0,2)-(1,3), C10: (2,0)-(3,1), C11: (2,2)-(3,3)
        C_blocks = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)]
        ]

        strassen_terms = [
            # m1 = (x00+x11)*(y00+y11)
            ({(0, 0): 1, (1, 1): 1},
             {(0, 0): 1, (1, 1): 1},
             {(0, 0): 1, (1, 1): 1}),
            # m2 = (x10+x11)*y00
            ({(1, 0): 1, (1, 1): 1},
             {(0, 0): 1},
             {(1, 0): 1, (1, 1): -1}),
            # m3 = x00*(y01-y11)
            ({(0, 0): 1},
             {(0, 1): 1, (1, 1): -1},
             {(0, 1): 1, (1, 1): 1}),
            # m4 = x11*(y10-y00)
            ({(1, 1): 1},
             {(1, 0): 1, (0, 0): -1},
             {(0, 0): 1, (1, 0): 1}),
            # m5 = (x00+x01)*y11
            ({(0, 0): 1, (0, 1): 1},
             {(1, 1): 1},
             {(0, 0): -1, (0, 1): 1}),
            # m6 = (x10-x00)*(y00+y01)
            ({(1, 0): 1, (0, 0): -1},
             {(0, 0): 1, (0, 1): 1},
             {(1, 1): 1}),
            # m7 = (x01-x11)*(y10+y11)
            ({(0, 1): 1, (1, 1): -1},
             {(1, 0): 1, (1, 1): 1},
             {(0, 0): 1}),
        ]
        
        for U_loc, V_loc, W_loc in strassen_terms:
            if time.time() > deadline:
                return
            
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            # Expand U_loc (coeffs for X entries) to A entries
            for (lr, lc), coeff_x in U_loc.items():
                for b_idx, coeff_a in X_coeffs.items():
                    if coeff_a == 0: continue
                    # X entry (lr, lc) is sum of A_block[b_idx][lr, lc]
                    # A_block[b_idx][lr, lc] is at global index
                    r, c = A_blocks[b_idx][lr * 2 + lc] # Wait, A_blocks stores list of 4 tuples.
                    # Let's fix A_blocks structure to be list of lists of (r,c)
                    # A_blocks[0] = [(0,0), (0,1), (1,0), (1,1)]
                    # Index lr*2+lc in the list corresponds to local position (lr, lc)?
                    # Yes, if we flatten row-major.
                    gr, gc = A_blocks[b_idx][lr * 2 + lc]
                    U_col[gr * m + gc] += coeff_x * coeff_a
            
            # Expand V_loc (coeffs for Y entries) to B entries
            for (lr, lc), coeff_y in V_loc.items():
                for b_idx, coeff_b in Y_coeffs.items():
                    if coeff_b == 0: continue
                    gr, gc = B_blocks[b_idx][lr * 2 + lc]
                    V_col[gr * p + gc] += coeff_y * coeff_b
            
            # Expand W_loc (coeffs for Z entries) to C entries
            # Z is the product matrix.
            # W_targets maps target C block index to coeff.
            for (lr, lc), coeff_z in W_loc.items():
                for c_block_idx, coeff_c in W_targets:
                    gr, gc = C_blocks[c_block_idx][lr * 2 + lc]
                    # W uses trace convention: W[k*n + i, r] where k is col, i is row of C
                    # C is 4x6. k is col index (0..5), i is row index (0..3).
                    W_col[gc * n + gr] += coeff_z * coeff_c
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Part 1: 4x4 * 4x4 (first 4 cols of B)
    # Outer Strassen terms for 4x4
    # A blocks: 0:00, 1:01, 2:10, 3:11
    # B blocks: 0:00, 1:01, 2:10, 3:11
    
    outer_strassen = [
        # (X_coeffs, Y_coeffs, W_targets)
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00 -> C10, C11
        ({2: 1, 3: 1}, {0: 1}, [(2, 1), (3, -1)]),
        # M3 = A00(B01-B11) -> C01, C11
        ({0: 1}, {1: 1, 3: -1}, [(1, 1), (3, 1)]),
        # M4 = A11(B10-B00) -> C00, C01, C10
        ({3: 1}, {2: 1, 0: -1}, [(0, -1), (1, 1), (2, 1)]),
        # M5 = (A00+A01)B11 -> C00, C11
        ({0: 1, 1: 1}, {3: 1}, [(0, 1), (3, 1)]),
        # M6 = (A10-A00)(B00+B01) -> C00, C10
        ({2: 1, 0: -1}, {0: 1, 1: 1}, [(0, 1), (2, 1)]),
        # M7 = (A01-A11)(B10+B11) -> C01, C11
        ({1: 1, 3: -1}, {2: 1, 3: 1}, [(1, 1), (3, 1)]),
    ]
    
    for X_c, Y_c, W_t in outer_strassen:
        add_block_strassen(X_c, Y_c, W_t)

    # Part 2: 4x4 * 4x2 (last 2 cols of B)
    # This is 4 independent 2x2 * 2x2 products.
    # A row blocks: 0 (rows 0-1), 1 (rows 2-3)
    # B col block: last 2 cols (cols 4-5).
    # We need to multiply A (4x4) by B_last (4x2).
    # C_last (4x2) = A * B_last.
    # C_last[0:2, 4:6] = A[0:2, 0:2] * B[0:2, 4:6] + A[0:2, 2:4] * B[2:4, 4:6]
    # C_last[2:4, 4:6] = A[2:4, 0:2] * B[0:2, 4:6] + A[2:4, 2:4] * B[2:4, 4:6]
    
    def add_simple_strassen(i0, j0, k0):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        A block at (i0, j0), B block at (j0, k0).
        """
        # Reuse the logic from the original code but adapted to append to global lists
        strassen_terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in strassen_terms:
            if time.time() > deadline:
                return
            
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Call for the 4 products
    add_simple_strassen(0, 0, 4) # A[0:2, 0:2] * B[0:2, 4:6]
    add_simple_strassen(0, 2, 4) # A[0:2, 2:4] * B[2:4, 4:6]
    add_simple_strassen(2, 0, 4) # A[2:4, 0:2] * B[0:2, 4:6]
    add_simple_strassen(2, 2, 4) # A[2:4, 2:4] * B[2:4, 4:6]

    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
