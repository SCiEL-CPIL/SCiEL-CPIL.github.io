DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 1, 0, 0, 1, -1, 0], # X00: M1, M3, M5, M6
        [0, 0, 1, 1, 0, 0, 1], # X10: M2, M6, M7
        [1, 1, 0, 0, 1, 0, 0], # X11: M1, M2, M4, M5
        [0, 0, 0, 0, 1, 0, 1]  # X01: M5, M7
    ]
    # Wait, I need to be careful with indices.
    # X is 2x2.
    # X00, X01, X10, X11.
    # My list above:
    # Row 0: X00. Coeffs for M1..M7.
    # M1: X00+X11 -> 1
    # M2: X10+X11 -> 0
    # M3: X00 -> 1
    # M4: X11 -> 0
    # M5: X00+X01 -> 1
    # M6: X10-X00 -> -1
    # M7: X01-X11 -> 0
    # So X00 row: [1, 0, 1, 0, 1, -1, 0]
    
    # X01:
    # M1: 0
    # M2: 0
    # M3: 0
    # M4: 0
    # M5: 1
    # M6: 0
    # M7: 1
    # So X01 row: [0, 0, 0, 0, 1, 0, 1]
    
    # X10:
    # M1: 0
    # M2: 1
    # M3: 0
    # M4: 0
    # M5: 0
    # M6: 1
    # M7: 0
    # So X10 row: [0, 1, 0, 0, 0, 1, 0]
    
    # X11:
    # M1: 1
    # M2: 1
    # M3: 0
    # M4: 1
    # M5: 0
    # M6: 0
    # M7: -1
    # So X11 row: [1, 1, 0, 1, 0, 0, -1]
    
    U_coeffs = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # Y coefficients
    # Y00:
    # M1: 1
    # M2: 1
    # M3: 0
    # M4: -1
    # M5: 0
    # M6: 1
    # M7: 0
    # Row: [1, 1, 0, -1, 0, 1, 0]
    
    # Y01:
    # M1: 0
    # M2: 0
    # M3: 1
    # M4: 0
    # M5: 0
    # M6: 1
    # M7: 0
    # Row: [0, 0, 1, 0, 0, 1, 0]
    
    # Y10:
    # M1: 0
    # M2: 0
    # M3: 0
    # M4: 1
    # M5: 0
    # M6: 0
    # M7: 1
    # Row: [0, 0, 0, 1, 0, 0, 1]
    
    # Y11:
    # M1: 1
    # M2: 0
    # M3: -1
    # M4: 0
    # M5: 1
    # M6: 0
    # M7: 1
    # Row: [1, 0, -1, 0, 1, 0, 1]
    
    V_coeffs = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W coefficients (Result Z)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # Z00
        [0, 0, 1, 0, 1, 0, 0],  # Z01
        [0, 1, 0, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    # Blocks:
    # 1. A_L (0,0:2,2) x B_LL (0,0:2,2) -> C (0,0:2,2)
    # 2. A_R (0,2:2,4) x B_RL (2,0:4,2) -> C (0,0:2,2)
    # 3. A_L (0,0:2,2) x B_LR (0,2:2,4) -> C (0,2:2,4)
    # 4. A_R (0,2:2,4) x B_RR (2,2:4,4) -> C (0,2:2,4)
    
    col_offset = 0
    for block in range(4):
        # Determine offsets for A, B, C
        if block == 0:
            a_i_off = 0; a_j_off = 0
            b_j_off = 0; b_k_off = 0
            c_i_off = 0; c_k_off = 0
        elif block == 1:
            a_i_off = 0; a_j_off = 2
            b_j_off = 2; b_k_off = 0
            c_i_off = 0; c_k_off = 0
        elif block == 2:
            a_i_off = 0; a_j_off = 0
            b_j_off = 0; b_k_off = 2
            c_i_off = 0; c_k_off = 2
        elif block == 3:
            a_i_off = 0; a_j_off = 2
            b_j_off = 2; b_k_off = 2
            c_i_off = 0; c_k_off = 2
            
        for r in range(7):
            col = col_offset + r
            # U part
            for i in range(2):
                for j in range(2):
                    coeff = U_coeffs[i*2+j][r]
                    if coeff != 0:
                        u_idx = (a_i_off + i) * 4 + (a_j_off + j)
                        U[u_idx][col] += coeff
                        
            # V part
            for j in range(2):
                for k in range(2):
                    coeff = V_coeffs[j*2+k][r]
                    if coeff != 0:
                        v_idx = (b_j_off + j) * 4 + (b_k_off + k)
                        V[v_idx][col] += coeff
                        
            # W part
            for i in range(2):
                for k in range(2):
                    coeff = W_coeffs[i*2+k][r]
                    if coeff != 0:
                        w_idx = (c_k_off + k) * 2 + (c_i_off + i)
                        W[w_idx][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
