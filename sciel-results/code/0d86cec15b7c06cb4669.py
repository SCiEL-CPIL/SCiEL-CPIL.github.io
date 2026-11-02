"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define Strassen terms for a single 2x2 multiplication X * Y = Z
    # X indices: (0,0), (0,1), (1,0), (1,1) relative to X
    # Y indices: (0,0), (0,1), (1,0), (1,1) relative to Y
    # Z indices: (0,0), (0,1), (1,0), (1,1) relative to Z
    # 
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y10-Y11)
    # M4: X11(Y01-Y10)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # List of (u_coeff, v_coeff, w_coeff) for each of the 7 terms
    # u: indices in X (0,1,4,5)
    # v: indices in Y (0,1,4,5)
    # w: indices in Z (0,1,2,3) -- NOTE: Z here is 2x2, row-major?
    # Let's check W index convention again.
    # W[k*n+i]. For a 2x2 block Z, if it maps to C rows 0,1 and cols k0, k0+1.
    # Z[0,0] -> C[i0, k0] -> W[k0*n + i0]
    # Z[0,1] -> C[i0, k0+1] -> W[(k0+1)*n + i0]
    # Z[1,0] -> C[i0+1, k0] -> W[k0*n + i0+1]
    # Z[1,1] -> C[i0+1, k0+1] -> W[(k0+1)*n + i0+1]
    
    # Strassen contributions to Z (local 2x2 indices 0,1,2,3 for (0,0),(0,1),(1,0),(1,1))
    # Term 1 (M1): +1 to Z00, +1 to Z11
    # Term 2 (M2): +1 to Z10, -1 to Z11
    # Term 3 (M3): +1 to Z01, +1 to Z11
    # Term 4 (M4): +1 to Z00, +1 to Z10
    # Term 5 (M5): -1 to Z00, +1 to Z01
    # Term 6 (M6): +1 to Z11
    # Term 7 (M7): +1 to Z00
    
    # Let's verify M6 and M7 signs from standard Strassen:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # M1: X(00,11) * Y(00,11) -> Z(00, 11) coeffs: 1, 1
    # M2: X(10,11) * Y(00) -> Z(10, 11) coeffs: 1, -1
    # M3: X(00) * Y(10, 11) -> Z(01, 11) coeffs: 1, 1  (Wait, Y10-Y11 means Y10 coeff 1, Y11 coeff -1)
    #    Actually, M3 = X00 * (Y10 - Y11). So V part has Y10:1, Y11:-1.
    #    Z part: Z01 += M3, Z11 += M3.
    # M4: X(11) * Y(01, 10) -> Z(00, 10) coeffs: 1, 1 (Y01-Y10 means Y01:1, Y10:-1)
    # M5: X(00, 01) * Y(11) -> Z(00, 01) coeffs: -1, 1
    # M6: X(10, 00) * Y(00, 01) -> Z(11) coeff: 1 (X10-X00 means X10:1, X00:-1)
    # M7: X(01, 11) * Y(10, 11) -> Z(00) coeff: 1 (X01-X11 means X01:1, X11:-1)

    # We define the 7 terms as tuples of (U_indices, U_coefs, V_indices, V_coefs, W_indices, W_coefs)
    # Local indices 0,1,4,5 for U and V (corresponding to 2x2 block entries)
    # Local indices 0,1,2,3 for W (corresponding to Z00, Z01, Z10, Z11)
    
    terms = [
        # M1
        ({0: 1, 5: 1}, {0: 1, 5: 1}, {0: 1, 3: 1}),
        # M2
        ({4: 1, 5: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {4: 1, 5: -1}, {1: 1, 3: 1}),
        # M4
        ({5: 1}, {1: 1, 4: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {5: 1}, {0: -1, 1: 1}),
        # M6
        ({4: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 5: -1}, {4: 1, 5: 1}, {0: 1}),
    ]

    # Define the 4 block multiplications
    # (U_base, V_base, C_row_offset, C_col_offset)
    # U_base: start index in U for the 2x2 block A
    # V_base: start index in V for the 2x2 block B
    # C_row_offset: row in C (0 or 1) -- actually C is 2x4, rows are 0,1.
    # C_col_offset: col in C (0, 1, 2, 3)
    
    # A is 2x4.
    # A00: rows 0,1, cols 0,1 -> U indices 0,1,4,5. Base 0.
    # A01: rows 0,1, cols 2,3 -> U indices 2,3,6,7. Base 2.
    
    # B is 4x4.
    # B00: rows 0,1, cols 0,1 -> V indices 0,1,4,5. Base 0.
    # B10: rows 2,3, cols 0,1 -> V indices 8,9,12,13. Base 8.
    # B01: rows 0,1, cols 2,3 -> V indices 2,3,6,7. Base 2.
    # B11: rows 2,3, cols 2,3 -> V indices 10,11,14,15. Base 10.
    
    # C is 2x4.
    # Block 1: A00 * B00 -> C[:, 0:2]. C_col_offset = 0.
    # Block 2: A01 * B10 -> C[:, 0:2]. C_col_offset = 0.
    # Block 3: A00 * B01 -> C[:, 2:4]. C_col_offset = 2.
    # Block 4: A01 * B11 -> C[:, 2:4]. C_col_offset = 2.
    
    blocks = [
        (0, 0, 0),   # A00, B00, Col 0
        (2, 8, 0),   # A01, B10, Col 0
        (0, 2, 2),   # A00, B01, Col 2
        (2, 10, 2),  # A01, B11, Col 2
    ]
    
    for u_base, v_base, c_col_off in blocks:
        for u_dict, v_dict, w_dict in terms:
            for u_rel, u_val in u_dict.items():
                U[u_base + u_rel][col_idx] += u_val
            for v_rel, v_val in v_dict.items():
                V[v_base + v_rel][col_idx] += v_val
            for w_rel, w_val in w_dict.items():
                # w_rel is 0,1,2,3 corresponding to Z00, Z01, Z10, Z11
                # Z00: C[0, c_col_off] -> W[(c_col_off)*2 + 0]
                # Z01: C[0, c_col_off+1] -> W[(c_col_off+1)*2 + 0]
                # Z10: C[1, c_col_off] -> W[(c_col_off)*2 + 1]
                # Z11: C[1, c_col_off+1] -> W[(c_col_off+1)*2 + 1]
                
                if w_rel == 0: # Z00
                    w_idx = c_col_off * 2 + 0
                elif w_rel == 1: # Z01
                    w_idx = (c_col_off + 1) * 2 + 0
                elif w_rel == 2: # Z10
                    w_idx = c_col_off * 2 + 1
                else: # Z11
                    w_idx = (c_col_off + 1) * 2 + 1
                
                W[w_idx][col_idx] += w_val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
