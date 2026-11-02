import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M = X Y
    # P1 = (X11+X22)(Y11+Y22)
    # P2 = (X21+X22)Y11
    # P3 = X11(Y12-Y22)
    # P4 = X22(Y21-Y11)
    # P5 = (X11+X12)Y22
    # P6 = (X21-X11)(Y11+Y12)
    # P7 = (X12-X22)(Y21+Y22)
    
    # X indices in 2x2 block: 00, 01, 10, 11 -> 0, 1, 2, 3 local
    # Y indices in 2x2 block: 00, 01, 10, 11 -> 0, 1, 2, 3 local
    # M indices in 2x2 block: 00, 01, 10, 11 -> 0, 1, 2, 3 local
    
    strassen_defs = [
        # (x_indices, x_coeffs, y_indices, y_coeffs, m_indices, m_coeffs)
        # P1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, 1]),
        # P3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # P4
        ([3], [1], [2, 0], [1, -1], [2, 3], [1, 1]),
        # P5
        ([0, 1], [1, 1], [3], [1], [0, 1], [1, 1]),
        # P6
        ([2, 0], [1, -1], [0, 1], [1, 1], [1, 3], [1, 1]),
        # P7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 3], [1, 1])
    ]
    
    # Define the 4 block multiplications
    # Block 1: A1 * B11 -> CL
    # A1 is A[:, 0:2], B11 is B[0:2, 0:2]
    # Block 2: A2 * B21 -> CL
    # A2 is A[:, 2:4], B21 is B[2:4, 0:2]
    # Block 3: A1 * B12 -> CR
    # A1 is A[:, 0:2], B12 is B[0:2, 2:4]
    # Block 4: A2 * B22 -> CR
    # A2 is A[:, 2:4], B22 is B[2:4, 2:4]
    
    blocks = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        # A is 2x4, B is 4x4, C is 2x4
        # A1: rows 0-1, cols 0-1
        # B11: rows 0-1, cols 0-1
        # C_L: rows 0-1, cols 0-1
        (0, 0, 0, 0, 0, 0),
        # A2: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1
        # C_L: rows 0-1, cols 0-1
        (0, 2, 2, 0, 0, 0),
        # A1: rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-3
        # C_R: rows 0-1, cols 2-3
        (0, 0, 0, 2, 0, 2),
        # A2: rows 0-1, cols 2-3
        # B22: rows 2-3, cols 2-3
        # C_R: rows 0-1, cols 2-3
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for s_idx, (x_ind, x_cof, y_ind, y_cof, m_ind, m_cof) in enumerate(strassen_defs):
            # Fill U
            for i, c in zip(x_ind, x_cof):
                # Local index i (0,1,2,3) maps to global A index
                # i=0 -> (0,0), i=1 -> (0,1), i=2 -> (1,0), i=3 -> (1,1)
                # Global A row = ar + i//2, Global A col = ac + i%2
                # Flat A index = (ar + i//2)*4 + (ac + i%2)
                a_row = ar + i // 2
                a_col = ac + i % 2
                u_idx = a_row * 4 + a_col
                U[u_idx, col] += c
            
            # Fill V
            for i, c in zip(y_ind, y_cof):
                # Local index i maps to global B index
                # Global B row = br + i//2, Global B col = bc + i%2
                # Flat B index = (br + i//2)*4 + (bc + i%2)
                b_row = br + i // 2
                b_col = bc + i % 2
                v_idx = b_row * 4 + b_col
                V[v_idx, col] += c
            
            # Fill W
            for i, c in zip(m_ind, m_cof):
                # Local index i maps to global C index
                # Global C row = cr + i//2, Global C col = cc + i%2
                # Task convention: W[k*n+i, r] where k is col, i is row
                # n=2. Index = col*2 + row
                c_row = cr + i // 2
                c_col = cc + i % 2
                w_idx = c_col * 2 + c_row
                W[w_idx, col] += c
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
