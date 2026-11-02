import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(g+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+b)g
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 6 blocks of the 2x4 x 4x6 multiplication.
    # A is 2x4, B is 4x6.
    # Split A into AL (cols 0-1) and AR (cols 2-3).
    # Split B into BT (rows 0-1) and BB (rows 2-3).
    # BT is 2x6, BB is 2x6.
    # Split BT into BT1 (cols 0-1), BT2 (cols 2-3), BT3 (cols 4-5).
    # Split BB into BB1 (cols 0-1), BB2 (cols 2-3), BB3 (cols 4-5).
    
    # Each block is a 2x2 x 2x2 multiplication.
    # Block (A_part, B_part) -> C_part (2x2)
    
    blocks = [
        # (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
        (0, 0, 0, 0, 0, 0), # AL * BT1 -> C[0:2, 0:2]
        (0, 0, 0, 2, 0, 2), # AL * BT2 -> C[0:2, 2:4]
        (0, 0, 0, 4, 0, 4), # AL * BT3 -> C[0:2, 4:6]
        (0, 2, 2, 0, 0, 0), # AR * BB1 -> C[0:2, 0:2]
        (0, 2, 2, 2, 0, 2), # AR * BB2 -> C[0:2, 2:4]
        (0, 2, 2, 4, 0, 4), # AR * BB3 -> C[0:2, 4:6]
    ]
    
    # Strassen terms definition
    # u: coefficients for A (local 2x2)
    # v: coefficients for B (local 2x2)
    # w: coefficients for C (local 2x2)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [[1, 0], [0, 1]] (a+d)
        # v: [[0, 0], [1, 1]] (g+h)
        # w: [[1, 0], [0, 1]] (C00, C11)
        ([[1, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 0], [0, 1]]),
        
        # M2: (c+d)h
        # u: [[0, 0], [1, 1]]
        # v: [[0, 0], [0, 1]]
        # w: [[0, 0], [1, 0]] (C10)
        ([[0, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [1, 0]]),
        
        # M3: a(f-h)
        # u: [[1, 0], [0, 0]]
        # v: [[0, 1], [0, -1]]
        # w: [[0, 1], [0, 1]] (C01, C11)
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        
        # M4: d(e-g)
        # u: [[0, 0], [0, 1]]
        # v: [[1, 0], [-1, 0]]
        # w: [[1, 0], [0, 0]] (C00)
        ([[0, 0], [0, 1]], [[1, 0], [-1, 0]], [[1, 0], [0, 0]]),
        
        # M5: (a+b)g
        # u: [[1, 1], [0, 0]]
        # v: [[0, 0], [1, 0]]
        # w: [[-1, 1], [0, 0]] (C00, C01)
        ([[1, 1], [0, 0]], [[0, 0], [1, 0]], [[-1, 1], [0, 0]]),
        
        # M6: (c-a)(e+g)
        # u: [[-1, 0], [1, 0]]
        # v: [[1, 0], [1, 0]]
        # w: [[0, 0], [1, 1]] (C10, C11)
        ([[-1, 0], [1, 0]], [[1, 0], [1, 0]], [[0, 0], [1, 1]]),
        
        # M7: (b-d)(f+h)
        # u: [[0, 1], [0, -1]]
        # v: [[0, 1], [0, 1]]
        # w: [[1, 0], [0, 0]] (C00)
        ([[0, 1], [0, -1]], [[0, 1], [0, 1]], [[1, 0], [0, 0]]),
    ]
    
    term_idx = 0
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (u_local, v_local, w_local) in strassen_terms:
            # Add this term to U, V, W
            for i in range(2):
                for j in range(2):
                    val_u = u_local[i][j]
                    if val_u != 0:
                        global_u_idx = (ar + i) * m + (ac + j)
                        U[global_u_idx, term_idx] += val_u
            
            for j in range(2):
                for k in range(2):
                    val_v = v_local[j][k]
                    if val_v != 0:
                        global_v_idx = (br + j) * p + (bc + k)
                        V[global_v_idx, term_idx] += val_v
            
            for k in range(2):
                for i in range(2):
                    val_w = w_local[i][k] # Note: w_local is indexed by [i][k] in C[i][k]
                    if val_w != 0:
                        global_w_idx = (cc + k) * n + (cr + i)
                        W[global_w_idx, term_idx] += val_w
            
            term_idx += 1
            
    return {"U": U, "V": V, "W": W}
