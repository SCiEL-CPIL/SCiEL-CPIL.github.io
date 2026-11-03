import time

DIMENSIONS = [2, 4, 4]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # x00, x01, x10, x11
    # y00, y01, y10, y11
    # z00, z01, z10, z11
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1: (x00+x11)(y00+y11) -> z00, z11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (x10+x11)y00 -> z10, -z11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: x00(y01-y11) -> z01, z11
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: x11(y10-y00) -> z00, z10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (x00+x01)y11 -> -z00, z01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (x10-x00)(y00+y01) -> z11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (x01-x11)(y10+y11) -> z00
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Block definitions:
    # Block 1: A_L (cols 0,1), B_TL (rows 0,1 cols 0,1), C_L (cols 0,1)
    # Block 2: A_R (cols 2,3), B_BL (rows 2,3 cols 0,1), C_L (cols 0,1)
    # Block 3: A_L (cols 0,1), B_TR (rows 0,1 cols 2,3), C_R (cols 2,3)
    # Block 4: A_R (cols 2,3), B_BR (rows 2,3 cols 2,3), C_R (cols 2,3)
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0), # Block 1
        (2, 2, 0, 0), # Block 2
        (0, 0, 2, 2), # Block 3
        (2, 2, 2, 2)  # Block 4
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            # u_coeffs keys are local indices 0..3 for x00, x01, x10, x11
            # Local 0 -> A[0, a_off] -> global idx 0*m + a_off
            # Local 1 -> A[0, a_off+1] -> global idx 0*m + a_off+1
            # Local 2 -> A[1, a_off] -> global idx 1*m + a_off
            # Local 3 -> A[1, a_off+1] -> global idx 1*m + a_off+1
            
            for local_idx, val in u_coeffs.items():
                if local_idx == 0:
                    g_idx = 0 * m + a_off
                elif local_idx == 1:
                    g_idx = 0 * m + a_off + 1
                elif local_idx == 2:
                    g_idx = 1 * m + a_off
                elif local_idx == 3:
                    g_idx = 1 * m + a_off + 1
                U[g_idx][col_idx] = val
            
            # Fill V
            # v_coeffs keys are local indices 0..3 for y00, y01, y10, y11
            # Local 0 -> B[b_row_off, b_col_off] -> global idx b_row_off*p + b_col_off
            # Local 1 -> B[b_row_off, b_col_off+1] -> global idx b_row_off*p + b_col_off+1
            # Local 2 -> B[b_row_off+1, b_col_off] -> global idx (b_row_off+1)*p + b_col_off
            # Local 3 -> B[b_row_off+1, b_col_off+1] -> global idx (b_row_off+1)*p + b_col_off+1
            
            for local_idx, val in v_coeffs.items():
                if local_idx == 0:
                    g_idx = b_row_off * p + b_col_off
                elif local_idx == 1:
                    g_idx = b_row_off * p + b_col_off + 1
                elif local_idx == 2:
                    g_idx = (b_row_off + 1) * p + b_col_off
                elif local_idx == 3:
                    g_idx = (b_row_off + 1) * p + b_col_off + 1
                V[g_idx][col_idx] = val
            
            # Fill W
            # w_coeffs keys are local indices 0..3 for z00, z01, z10, z11
            # Local 0 -> C[0, c_off] -> global idx c_off*n + 0
            # Local 1 -> C[0, c_off+1] -> global idx (c_off+1)*n + 0
            # Local 2 -> C[1, c_off] -> global idx c_off*n + 1
            # Local 3 -> C[1, c_off+1] -> global idx (c_off+1)*n + 1
            
            for local_idx, val in w_coeffs.items():
                if local_idx == 0:
                    g_idx = c_off * n + 0
                elif local_idx == 1:
                    g_idx = (c_off + 1) * n + 0
                elif local_idx == 2:
                    g_idx = c_off * n + 1
                elif local_idx == 3:
                    g_idx = (c_off + 1) * n + 1
                W[g_idx][col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
