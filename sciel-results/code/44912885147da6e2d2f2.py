"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00       -> C10
    # M3: A00(B01-B11)       -> C01
    # M4: A11(B10-B00)       -> C00, C10
    # M5: (A00+A01)B11       -> -C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: A00, A01, A10, A11 (indices 0,1,2,3)
        # V: B00, B01, B10, B11 (indices 0,1,2,3)
        # W: C00, C01, C10, C11 (indices 0,1,2,3)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),               # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),              # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # M7
    ]
    
    col = 0
    # Blocks: (A_row_offset, A_col_offset, B_row_offset, B_col_offset)
    # A is 2x4. B is 4x6.
    # A_L: rows 0-1, cols 0-1. A_R: rows 0-1, cols 2-3.
    # B_11: rows 0-1, cols 0-1. B_12: rows 0-1, cols 2-3. B_13: rows 0-1, cols 4-5.
    # B_21: rows 2-3, cols 0-1. B_22: rows 2-3, cols 2-3. B_23: rows 2-3, cols 4-5.
    
    blocks = [
        (0, 0, 0, 0), # A_L B_11
        (0, 0, 0, 2), # A_L B_12
        (0, 0, 0, 4), # A_L B_13
        (0, 2, 2, 0), # A_R B_21
        (0, 2, 2, 2), # A_R B_22
        (0, 2, 2, 4), # A_R B_23
    ]
    
    for a_row, a_col, b_row, b_col in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for idx, val in u_coeffs.items():
                # Local A index idx -> Global
                # A is 2x4. Local 2x2.
                # Local i = idx // 2, j = idx % 2
                li = idx // 2
                lj = idx % 2
                gi = a_row + li
                gj = a_col + lj
                U[gi * m + gj][col] += val
            for idx, val in v_coeffs.items():
                # Local B index idx -> Global
                # B is 4x6. Local 2x2.
                # Local i = idx // 2, j = idx % 2
                li = idx // 2
                lj = idx % 2
                gi = b_row + li
                gj = b_col + lj
                V[gi * p + gj][col] += val
            for idx, val in w_coeffs.items():
                # Local C index idx -> Global
                # C is 2x6. Local 2x2.
                # Local i = idx // 2, j = idx % 2
                li = idx // 2
                lj = idx % 2
                gi = a_row + li # C row matches A row
                gj = b_col + lj # C col matches B col
                # W index is k*n + i. Here k is col, i is row.
                # Global C row gi, col gj.
                # Index = gj * n + gi
                W[gj * n + gi][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
