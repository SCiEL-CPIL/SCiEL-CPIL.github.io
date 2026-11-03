def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # Block 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A indices: 0,1 (cols) -> global A index i*m + j
    # B indices: 0,1 (rows) -> global B index j*p + k
    # C indices: all cols
    
    # We define a function to add a 2x2x5 block
    def add_block(a_col_offset, b_row_offset, w_offset):
        # w_offset is the starting column in U,V,W for this block (0 or 18)
        
        # 18 multiplications
        # 0-6: Strassen for Y cols 0,1
        # 7-13: Strassen for Y cols 2,3
        # 14-17: Schoolbook for Y col 4
        
        # Helper to add a multiplication term
        # term: (coeffs for A, coeffs for B, coeffs for C)
        # A coeffs: list of (global_a_index, coeff)
        # B coeffs: list of (global_b_index, coeff)
        # C coeffs: list of (global_c_index, coeff)
        
        def add_term(local_r, a_coeffs, b_coeffs, c_coeffs):
            r = w_offset + local_r
            for idx, val in a_coeffs:
                U[idx][r] = val
            for idx, val in b_coeffs:
                V[idx][r] = val
            for idx, val in c_coeffs:
                W[idx][r] = val

        # Strassen for Y1 (local cols 0,1)
        # X = A[:, a_col_offset:a_col_offset+2]
        # Y = B[b_row_offset:b_row_offset+2, 0:2]
        
        # M1: (x00+x11)(y00+y11)
        a00 = a_col_offset
        a01 = a_col_offset + 1
        a10 = a_col_offset
        a11 = a_col_offset + 1
        
        b00 = b_row_offset
        b01 = b_row_offset + 1
        
        # Global A indices: i*m + j
        # i=0, j=a00 -> 0*m + a00
        # i=1, j=a10 -> 1*m + a10
        # etc.
        
        # M1
        add_term(0, 
                 [(0*m + a00, 1), (1*m + a11, 1)], 
                 [(b00*p + 0, 1), (b01*p + 1, 1)], 
                 [(0*n + 0, 1), (1*n + 1, 1)]) # C00, C11
        
        # M2: (x10+x11)y00
        add_term(1, 
                 [(1*m + a10, 1), (1*m + a11, 1)], 
                 [(b00*p + 0, 1)], 
                 [(1*n + 0, 1), (1*n + 1, 1)]) # C10, C11? 
        # Strassen: Z21 = M2 + M4. Z22 = M1 - M2 + M3 + M6.
        # My mapping:
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # So M2 contributes to Z10 (coeff 1) and Z11 (coeff -1)
        add_term(1, 
                 [(1*m + a10, 1), (1*m + a11, 1)], 
                 [(b00*p + 0, 1)], 
                 [(1*n + 0, 1), (1*n + 1, -1)])
        
        # M3: x00(y01-y11)
        add_term(2, 
                 [(0*m + a00, 1)], 
                 [(b00*p + 1, 1), (b01*p + 1, -1)], 
                 [(0*n + 1, 1), (1*n + 1, 1)])
        
        # M4: x11(y10-y00)
        add_term(3, 
                 [(1*m + a11, 1)], 
                 [(b01*p + 0, 1), (b00*p + 0, -1)], 
                 [(0*n + 0, 1), (1*n + 0, 1)])
        
        # M5: (x00+x01)y11
        add_term(4, 
                 [(0*m + a00, 1), (0*m + a01, 1)], 
                 [(b01*p + 1, 1)], 
                 [(0*n + 0, -1), (0*n + 1, 1)])
        
        # M6: (x10-x00)(y00+y01)
        add_term(5, 
                 [(1*m + a10, 1), (0*m + a00, -1)], 
                 [(b00*p + 0, 1), (b00*p + 1, 1)], 
                 [(1*n + 1, 1)])
        
        # M7: (x01-x11)(y10+y11)
        add_term(6, 
                 [(0*m + a01, 1), (1*m + a11, -1)], 
                 [(b01*p + 0, 1), (b01*p + 1, 1)], 
                 [(0*n + 0, 1)])
        
        # Strassen for Y2 (local cols 2,3)
        # Y2 is B[b_row_offset:b_row_offset+2, 2:4]
        # M8-M14 (local 7-13)
        
        # M8: (x00+x11)(y20+y31) -> (x00+x11)(B[b0,2]+B[b1,3])
        add_term(7, 
                 [(0*m + a00, 1), (1*m + a11, 1)], 
                 [(b00*p + 2, 1), (b01*p + 3, 1)], 
                 [(0*n + 2, 1), (1*n + 3, 1)])
                 
        # M9: (x10+x11)y20
        add_term(8, 
                 [(1*m + a10, 1), (1*m + a11, 1)], 
                 [(b00*p + 2, 1)], 
                 [(1*n + 2, 1), (1*n + 3, -1)])
                 
        # M10: x00(y21-y31)
        add_term(9, 
                 [(0*m + a00, 1)], 
                 [(b00*p + 3, 1), (b01*p + 3, -1)], 
                 [(0*n + 3, 1), (1*n + 3, 1)])
                 
        # M11: x11(y30-y20)
        add_term(10, 
                 [(1*m + a11, 1)], 
                 [(b01*p + 2, 1), (b00*p + 2, -1)], 
                 [(0*n + 2, 1), (1*n + 2, 1)])
                 
        # M12: (x00+x01)y31
        add_term(11, 
                 [(0*m + a00, 1), (0*m + a01, 1)], 
                 [(b01*p + 3, 1)], 
                 [(0*n + 2, -1), (0*n + 3, 1)])
                 
        # M13: (x10-x00)(y20+y21)
        add_term(12, 
                 [(1*m + a10, 1), (0*m + a00, -1)], 
                 [(b00*p + 2, 1), (b00*p + 3, 1)], 
                 [(1*n + 3, 1)])
                 
        # M14: (x01-x11)(y30+y31)
        add_term(13, 
                 [(0*m + a01, 1), (1*m + a11, -1)], 
                 [(b01*p + 2, 1), (b01*p + 3, 1)], 
                 [(0*n + 2, 1)])
                 
        # Schoolbook for Y3 (local col 4)
        # Y3 is B[b_row_offset:b_row_offset+2, 4:5]
        # Z04 = x00*y40 + x01*y41
        # Z14 = x10*y40 + x11*y41
        
        # M15: x00*y40
        add_term(14, 
                 [(0*m + a00, 1)], 
                 [(b00*p + 4, 1)], 
                 [(0*n + 4, 1)])
                 
        # M16: x01*y41
        add_term(15, 
                 [(0*m + a01, 1)], 
                 [(b01*p + 4, 1)], 
                 [(0*n + 4, 1)])
                 
        # M17: x10*y40
        add_term(16, 
                 [(1*m + a10, 1)], 
                 [(b00*p + 4, 1)], 
                 [(1*n + 4, 1)])
                 
        # M18: x11*y41
        add_term(17, 
                 [(1*m + a11, 1)], 
                 [(b01*p + 4, 1)], 
                 [(1*n + 4, 1)])

    # Call for Block 1
    add_block(0, 0, 0)
    # Call for Block 2
    add_block(2, 2, 18)
    
    return {"U": U, "V": V, "W": W}
