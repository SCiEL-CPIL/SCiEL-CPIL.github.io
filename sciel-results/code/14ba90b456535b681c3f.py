import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Part 1: M_{1,4,4} for row 0 of A
    # Rank 16
    # Part 2: M_{2,4,4} for rows 1,2 of A
    # Rank 28 (4 * 7)
    
    total_rank = 16 + 28
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 16 terms for A[0, :] * B
    # C[0, k] = sum_j A[0, j] B[j, k]
    for k in range(4):
        for j in range(4):
            U[0 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # Part 2: 28 terms for A[1:3, :] * B
    # A is 2x4, B is 4x4, C is 2x4
    # Split B into 4 2x2 blocks
    # B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # A_L (cols 0:2), A_R (cols 2:4)
    
    # Strassen coefficients for 2x2
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Define the 7 terms for a generic 2x2 multiplication X*Y=Z
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    # Z indices: 00, 01, 10, 11
    
    strassen_terms = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # X_coeffs is dict {idx: coeff}
        # Y_coeffs is dict {idx: coeff}
        # Z_coeffs is dict {idx: coeff}
        
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # Wait, Z21 = M2 + M4, Z22 = ... - M2
        # Let's re-verify Z mapping for M2
        # Z21 = M2 + M4 -> M2 coeff 1 in Z21
        # Z22 = M1 - M2 + M3 + M6 -> M2 coeff -1 in Z22
        # So Z_coeffs for M2: {(1,0): 1, (1,1): -1}
        
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # Z12 = M3 + M5, Z22 = ... + M3
        # Z12 = M3 + M5 -> M3 coeff 1 in Z12
        # Z22 = M1 - M2 + M3 + M6 -> M3 coeff 1 in Z22
        # So Z_coeffs for M3: {(0,1): 1, (1,1): 1}
        
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # Z11 = M1 + M4 ..., Z21 = M2 + M4
        # Z11 = M1 + M4 - M5 + M7 -> M4 coeff 1 in Z11
        # Z21 = M2 + M4 -> M4 coeff 1 in Z21
        # So Z_coeffs for M4: {(0,0): 1, (1,0): 1}
        
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # Z11 = ... - M5, Z12 = M3 + M5
        # Z11 = M1 + M4 - M5 + M7 -> M5 coeff -1 in Z11
        # Z12 = M3 + M5 -> M5 coeff 1 in Z12
        # So Z_coeffs for M5: {(0,0): -1, (0,1): 1}
        
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # Z22 = ... + M6
        # Z22 = M1 - M2 + M3 + M6 -> M6 coeff 1 in Z22
        # So Z_coeffs for M6: {(1,1): 1}
        
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}), # Z11 = ... + M7
        # Z11 = M1 + M4 - M5 + M7 -> M7 coeff 1 in Z11
        # So Z_coeffs for M7: {(0,0): 1}
    ]
    
    # Blocks to process:
    # 1. A_L * B00 -> C_L0 (Rows 1,2 of C, Cols 0,1)
    # 2. A_L * B01 -> C_L1 (Rows 1,2 of C, Cols 2,3)
    # 3. A_R * B10 -> C_R0 (Rows 1,2 of C, Cols 0,1)
    # 4. A_R * B11 -> C_R1 (Rows 1,2 of C, Cols 2,3)
    
    # A_L: Rows 1,2 of A. Cols 0,1.
    # A_R: Rows 1,2 of A. Cols 2,3.
    
    blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        (1, 0, 0, 0, 1, 0), # A_L * B00 -> C_L0
        (1, 0, 0, 2, 1, 2), # A_L * B01 -> C_L1
        (1, 2, 2, 0, 1, 0), # A_R * B10 -> C_R0
        (1, 2, 2, 2, 1, 2), # A_R * B11 -> C_R1
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            # Add term
            for (xi, yi), ux in x_coeffs.items():
                # A index: row ar+xi, col ac+yi
                U[(ar + xi) * 4 + (ac + yi)][col] += ux
            
            for (yi, xi), vy in y_coeffs.items():
                # B index: row br+yi, col bc+xi
                # Note: In strassen_terms, keys are (row, col) of Y?
                # My keys were (0,0) etc. Let's assume (row, col).
                # In the tuple above, I used (0,0) for Y11.
                # Y11 is row 0, col 0.
                V[(br + yi) * 4 + (bc + xi)][col] += vy
            
            for (zi, zj), wz in z_coeffs.items():
                # C index: row cr+zi, col cc+zj
                # W index: k*n + i. Here k is col of C, i is row of C.
                # So W[(cc+zj)*3 + (cr+zi)]
                W[(cc + zj) * 3 + (cr + zi)][col] += wz
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
