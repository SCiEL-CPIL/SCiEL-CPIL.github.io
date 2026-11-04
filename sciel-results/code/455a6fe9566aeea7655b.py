"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 decomposition for 2x2x2 multiplication
    # For matrices A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = A*B, the 7 rank-1 terms correspond to:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    strassen_coeffs = [
        # Each entry: (U_factor, V_factor, W_factor)
        # m1 = (a11+a22)(b11+b22)
        (1, 1, 1),   # a11 * b11 -> contributes to C11 (via m1)
        (1, 2, 2),   # a11 * b22 -> contributes to C11 (via m1)
        (2, 1, 2),   # a22 * b11 -> contributes to C11 (via m1)
        (2, 2, 1),   # a22 * b22 -> contributes to C11 (via m1)
        # m2 = (a21+a22)b11
        (1, 1, 1),   # a21 * b11 -> contributes to C21 (via m2)
        (2, 1, 1),   # a22 * b11 -> contributes to C21 (via m2)
        # m3 = a11(b12-b22)
        (1, 2, 2),   # a11 * b12 -> contributes to C12 (via m3)
        (1, 1, 1),   # a11 * b22 -> contributes to C12 (via m3, negative)
        # m4 = a22(b21-b11)
        (2, 1, 2),   # a22 * b21 -> contributes to C21 (via m4)
        (2, 2, 1),   # a22 * b11 -> contributes to C21 (via m4, negative)
        # m5 = (a11+a12)b22
        (1, 1, 1),   # a11 * b22 -> contributes to C11 (via m5)
        (1, 2, 2),   # a12 * b22 -> contributes to C11 (via m5)
        # m6 = (a21-a11)(b11+b12)
        (1, 1, 1),   # a21 * b11 -> contributes to C22 (via m6)
        (1, 2, 2),   # a21 * b12 -> contributes to C22 (via m6)
        (2, 1, 1),   # a11 * b11 -> contributes to C22 (via m6, negative)
        (2, 2, 2),   # a11 * b12 -> contributes to C22 (via m6, negative)
        # m7 = (a12-a22)(b21+b22)
        (1, 1, 2),   # a12 * b21 -> contributes to C11 (via m7)
        (1, 2, 1),   # a12 * b22 -> contributes to C11 (via m7)
        (2, 1, 2),   # a22 * b21 -> contributes to C11 (via m7, negative)
        (2, 2, 1),   # a22 * b22 -> contributes to C11 (via m7, negative)
    ]
    
    # Actually, let me use a cleaner approach: enumerate the 7 Strassen products
    # directly as rank-1 tensors in the 2x2x2 space
    
    # For 2x2x2, the 7 rank-1 terms are:
    # T1: U=(1,0,1,0), V=(1,0,0,1), W=(1,0,0,1)  -> (a11+a22)*(b11+b22)
    # T2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,0)  -> (a21+a22)*b11
    # T3: U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,0)  -> a11*(b12-b22)
    # T4: U=(0,0,0,1), V=(0,-1,1,0), W=(0,0,1,0)  -> a22*(b21-b11)
    # T5: U=(1,1,0,0), V=(0,0,0,1), W=(1,0,0,0)  -> (a11+a12)*b22
    # T6: U=(0,-1,1,0), V=(1,1,0,0), W=(0,0,0,1)  -> (a21-a11)*(b11+b12)
    # T7: U=(0,1,0,-1), V=(0,0,1,1), W=(1,0,0,0)  -> (a12-a22)*(b21+b22)
    
    strassen_2x2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs) for 2x2x2
        # U is 4-dim (2x2 flattened row-major), V is 4-dim, W is 4-dim
        ((1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1)),   # T1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),   # T2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),  # T3
        ((0, 0, 0, 1), (0, -1, 1, 0), (0, 0, 1, 0)),  # T4
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0)),   # T5
        ((0, -1, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),  # T6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),  # T7
    ]
    
    col = 0
    for block in range(2):  # A1, A2 (each 2x2)
        for subcol in range(4):  # B blocks: B11, B12, B13, B14 (each 2x2)
            for s_idx in range(7):
                U_c, V_c, W_c = strassen_2x2x2[s_idx]
                # U_c: (a11, a12, a21, a22) -> rows 0,1 of A block, cols 0,1
                # V_c: (b11, b12, b21, b22) -> rows 0,1 of B, cols 2*subcol, 2*subcol+1
                # W_c: (c11, c12, c21, c22) -> rows 0,1 of C, cols 2*subcol, 2*subcol+1
                
                # U: A is 2x4, block starts at row 0, col 2*block
                for u_row in range(2):
                    for u_col in range(2):
                        a_idx = u_row * 4 + (2 * block + u_col)  # row u_row, col 2*block+u_col
                        U[a_idx][col] = U_c[u_row * 2 + u_col]
                
                # V: B is 4x8, subcol block starts at row 2*block (since A splits cols, B splits rows)
                # Wait, A is 2x4, B is 4x8. A = [A1 A2] where A1,A2 are 2x2 (cols 0-1, 2-3)
                # B should be split as B = [B1; B2] where B1 is 2x8 (rows 0-1), B2 is 2x8 (rows 2-3)
                # So for A1 (cols 0-1 of A), we multiply with B1 (rows 0-1 of B)
                # For A2 (cols 2-3 of A), we multiply with B2 (rows 2-3 of B)
                for v_row in range(2):
                    for v_col in range(2):
                        b_row = 2 * block + v_row  # rows 0-1 for block=0, rows 2-3 for block=1
                        b_col = 2 * subcol + v_col
                        b_idx = b_row * 8 + b_col
                        V[b_idx][col] = V_c[v_row * 2 + v_col]
                
                # W: C is 2x8, col 2*subcol + v_col
                for w_row in range(2):
                    for w_col in range(2):
                        c_row = w_row
                        c_col = 2 * subcol + w_col
                        c_idx = c_row * 8 + c_col
                        W[c_idx][col] = W_c[w_row * 2 + w_col]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

