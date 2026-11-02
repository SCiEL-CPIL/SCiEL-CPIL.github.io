import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 - M4 + M6 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M2 - M5
    
    strassen_x = [
        [(0,0), (1,1)], # M1
        [(1,0), (1,1)], # M2
        [(0,0)],        # M3
        [(1,1)],        # M4
        [(0,0), (0,1)], # M5
        [(1,0), (0,0)], # M6 (10 - 00) -> coeffs 1, -1
        [(0,1), (1,1)]  # M7 (01 - 11) -> coeffs 1, -1
    ]
    # Wait, M6 and M7 have negative coefficients in X.
    # M6 = (X10 - X00) ...
    # So U[10] = 1, U[00] = -1.
    
    strassen_y = [
        [(0,0), (1,1)], # M1
        [(0,0)],        # M2
        [(0,1), (1,1)], # M3 (Y01 - Y11) -> 1, -1
        [(1,0), (0,0)], # M4 (Y10 - Y00) -> 1, -1
        [(1,1)],        # M5
        [(0,0), (0,1)], # M6
        [(1,0), (1,1)]  # M7
    ]
    
    strassen_w = [
        {(0,0): 1, (1,1): 1, (1,0): -1, (0,1): -1}, # Z00: M1, Z11: M1, Z10: -M4? No.
        # Z00 = M1 - M4 + M6 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M2 - M5
        
        # Let's map M_s to Z_c coefficients
        # M1: Z00=1, Z11=1
        # M2: Z10=1, Z11=1
        # M3: Z01=1, Z11=-1
        # M4: Z00=-1, Z10=1
        # M5: Z01=1, Z11=-1
        # M6: Z00=1
        # M7: Z00=-1
    ]
    
    # Correct W coefficients:
    # s=1 (M1): Z00=1, Z11=1
    # s=2 (M2): Z10=1, Z11=1
    # s=3 (M3): Z01=1, Z11=-1
    # s=4 (M4): Z00=-1, Z10=1
    # s=5 (M5): Z01=1, Z11=-1
    # s=6 (M6): Z00=1
    # s=7 (M7): Z00=-1
    
    w_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(0,0): -1, (1,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,0): 1},
        {(0,0): -1}
    ]
    
    # U coeffs for X
    u_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # V coeffs for Y
    v_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    def add_block(j_offset, k_offset, col_offset):
        # j_offset: 0 or 2
        # k_offset: 0 or 2
        for s in range(7):
            r = col_offset + s
            # U
            for (i, j), val in u_coeffs[s].items():
                U[i * m + (j + j_offset), r] = val
            # V
            for (j, k), val in v_coeffs[s].items():
                V[(j + j_offset) * p + (k + k_offset), r] = val
            # W
            for (k, i), val in w_coeffs[s].items():
                W[(k + k_offset) * n + i, r] = val

    # Block 1: j in {0,1}, k in {0,1} and {2,3}
    # Wait, M_{2,2,4} splits k into two 2x2 blocks.
    # So for Block 1 (j=0,1), we have k=0,1 and k=2,3.
    # This requires 14 ranks.
    # Block 1, Part 1: j=0,1, k=0,1. Col 0-6.
    add_block(0, 0, 0)
    # Block 1, Part 2: j=0,1, k=2,3. Col 7-13.
    add_block(0, 2, 7)
    
    # Block 2: j in {2,3}
    # Block 2, Part 1: j=2,3, k=0,1. Col 14-20.
    add_block(2, 0, 14)
    # Block 2, Part 2: j=2,3, k=2,3. Col 21-27.
    add_block(2, 2, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
