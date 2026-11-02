import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for u (X part) and v (Y part)
    # u_s is linear combo of x_00, x_01, x_10, x_11
    # v_s is linear combo of y_00, y_01, y_10, y_11
    # Indices for X: (0,0), (0,1), (1,0), (1,1)
    # Indices for Y: (0,0), (0,1), (1,0), (1,1)
    
    # u coefficients: dict {step: {x_idx: coeff}}
    u_coeffs = [
        {(0,0): 1, (1,1): 1},       # P1: x00 + x11
        {(1,0): 1, (1,1): 1},       # P2: x10 + x11
        {(0,0): 1},                 # P3: x00
        {(1,1): 1},                 # P4: x11
        {(0,0): 1, (0,1): 1},       # P5: x00 + x01
        {(1,0): 1, (0,0): -1},      # P6: x10 - x00
        {(0,1): 1, (1,1): -1}       # P7: x01 - x11
    ]
    
    # v coefficients: dict {step: {y_idx: coeff}}
    v_coeffs = [
        {(0,0): 1, (1,1): 1},       # P1: y00 + y11
        {(0,0): 1},                 # P2: y00
        {(0,1): 1, (1,1): -1},      # P3: y01 - y11
        {(1,0): 1, (0,0): -1},      # P4: y10 - y00
        {(1,1): 1},                 # P5: y11
        {(0,0): 1, (0,1): 1},       # P6: y00 + y01
        {(1,0): 1, (1,1): 1}        # P7: y10 + y11
    ]
    
    # W coefficients: contribution of P_s to Z_00, Z_01, Z_10, Z_11
    # Z_00 = P1 + P4 - P5 + P7
    # Z_01 = P3 + P5
    # Z_10 = P2 + P4
    # Z_11 = P1 - P2 + P3 + P6
    # w_coeffs[step][z_idx] = coeff
    w_coeffs = [
        {(0,0): 1, (1,1): 1},       # P1
        {(1,0): 1, (1,1): -1},      # P2
        {(0,1): 1, (1,1): 1},       # P3
        {(0,0): 1, (1,0): 1},       # P4
        {(0,0): -1, (0,1): 1},      # P5
        {(1,1): 1},                 # P6
        {(0,0): 1}                  # P7
    ]
    # Wait, let's re-verify Z_11.
    # Z_11 = P1 - P2 + P3 + P6
    # P1: (1,1) += 1
    # P2: (1,1) += -1
    # P3: (1,1) += 1
    # P6: (1,1) += 1
    # So w_coeffs[0][(1,1)] += 1
    # w_coeffs[1][(1,1)] += -1
    # w_coeffs[2][(1,1)] += 1
    # w_coeffs[5][(1,1)] += 1
    
    # Let's build w_coeffs properly.
    w_coeffs = [{} for _ in range(7)]
    # P1
    w_coeffs[0][(0,0)] = w_coeffs[0].get((0,0), 0) + 1
    w_coeffs[0][(1,1)] = w_coeffs[0].get((1,1), 0) + 1
    # P2
    w_coeffs[1][(1,0)] = w_coeffs[1].get((1,0), 0) + 1
    w_coeffs[1][(1,1)] = w_coeffs[1].get((1,1), 0) - 1
    # P3
    w_coeffs[2][(0,1)] = w_coeffs[2].get((0,1), 0) + 1
    w_coeffs[2][(1,1)] = w_coeffs[2].get((1,1), 0) + 1
    # P4
    w_coeffs[3][(0,0)] = w_coeffs[3].get((0,0), 0) + 1
    w_coeffs[3][(1,0)] = w_coeffs[3].get((1,0), 0) + 1
    # P5
    w_coeffs[4][(0,0)] = w_coeffs[4].get((0,0), 0) - 1
    w_coeffs[4][(0,1)] = w_coeffs[4].get((0,1), 0) + 1
    # P6
    w_coeffs[5][(1,1)] = w_coeffs[5].get((1,1), 0) + 1
    # P7
    w_coeffs[6][(0,0)] = w_coeffs[6].get((0,0), 0) + 1
    
    col = 0
    # Part L: A_L (cols 0,1), B_T (rows 0,1)
    # Part R: A_R (cols 2,3), B_B (rows 2,3)
    for part in ['L', 'R']:
        # X indices
        if part == 'L':
            x_cols = [0, 1]
            y_rows = [0, 1]
        else:
            x_cols = [2, 3]
            y_rows = [2, 3]
            
        # 3 sub-blocks of Y (2x2 each)
        for t in range(3):
            # Y columns
            y_cols = [2*t, 2*t+1]
            
            for s in range(7):
                # Fill U
                for (xi, xj), coeff in u_coeffs[s].items():
                    # X index (xi, xj) corresponds to A row xi, col x_cols[xj]
                    u_row = xi * m + x_cols[xj]
                    U[u_row][col] = coeff
                
                # Fill V
                for (yi, yj), coeff in v_coeffs[s].items():
                    # Y index (yi, yj) corresponds to B row y_rows[yi], col y_cols[yj]
                    v_row = y_rows[yi] * p + y_cols[yj]
                    V[v_row][col] = coeff
                
                # Fill W
                for (zi, zj), coeff in w_coeffs[s].items():
                    # Z index (zi, zj) corresponds to C row zi, col y_cols[zj]
                    # W index is k*n + i. Here k is the column index in C, i is row index.
                    # C col is y_cols[zj]. C row is zi.
                    w_row = y_cols[zj] * n + zi
                    W[w_row][col] = coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
