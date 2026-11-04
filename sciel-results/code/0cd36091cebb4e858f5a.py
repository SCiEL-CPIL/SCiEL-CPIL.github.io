import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 6 groups
    # Each group is (A_start_row, A_start_col, B_start_row, B_start_col, C_start_row, C_start_col)
    # A is 2x4, B is 4x6, C is 2x6
    
    # A1: rows 0-1, cols 0-1
    # A2: rows 0-1, cols 2-3
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B13: rows 0-1, cols 4-5
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    # B23: rows 2-3, cols 4-5
    # C1: rows 0-1, cols 0-1
    # C2: rows 0-1, cols 2-3
    # C3: rows 0-1, cols 4-5
    
    groups = [
        (0, 0, 0, 0, 0, 0), # A1, B11 -> C1
        (0, 0, 0, 2, 0, 2), # A1, B12 -> C2
        (0, 0, 0, 4, 0, 4), # A1, B13 -> C3
        (0, 2, 2, 0, 0, 0), # A2, B21 -> C1
        (0, 2, 2, 2, 0, 2), # A2, B22 -> C2
        (0, 2, 2, 4, 0, 4), # A2, B23 -> C3
    ]
    
    col_offset = 0
    for (ar, ac, br, bc, cr, cc) in groups:
        # A block indices in U
        # A is stored row-major: index = i*m + j
        # A block is 2x2 starting at (ar, ac)
        # U indices:
        u_00 = ar * m + ac
        u_01 = ar * m + (ac + 1)
        u_10 = (ar + 1) * m + ac
        u_11 = (ar + 1) * m + (ac + 1)
        
        # B block indices in V
        # B is stored row-major: index = j*p + k
        # B block is 2x2 starting at (br, bc)
        # V indices:
        v_00 = br * p + bc
        v_01 = br * p + (bc + 1)
        v_10 = (br + 1) * p + bc
        v_11 = (br + 1) * p + (bc + 1)
        
        # C block indices in W
        # C is stored row-major: index = k*n + i
        # C block is 2x2 starting at (cr, cc)
        # W indices:
        w_00 = cc * n + cr
        w_01 = (cc + 1) * n + cr
        w_10 = cc * n + (cr + 1)
        w_11 = (cc + 1) * n + (cr + 1)
        
        # Strassen coefficients
        # M1 = (X00 + X11)(Y00 + Y11)
        # M2 = (X10 + X11)Y00
        # M3 = X00(Y01 - Y11)
        # M4 = X11(Y10 - X10)
        # M5 = (X00 + X01)Y11
        # M6 = (X10 - X00)(Y00 + Y01)
        # M7 = (X01 - X11)(Y10 + Y11)
        
        # U contributions
        U[u_00][col_offset + 0] += 1
        U[u_11][col_offset + 0] += 1
        U[u_10][col_offset + 1] += 1
        U[u_11][col_offset + 1] += 1
        U[u_00][col_offset + 2] += 1
        U[u_11][col_offset + 3] += 1
        U[u_10][col_offset + 3] += -1
        U[u_00][col_offset + 4] += 1
        U[u_01][col_offset + 4] += 1
        U[u_10][col_offset + 5] += 1
        U[u_00][col_offset + 5] += -1
        U[u_01][col_offset + 6] += 1
        U[u_11][col_offset + 6] += -1
        
        # V contributions
        V[v_00][col_offset + 0] += 1
        V[v_11][col_offset + 0] += 1
        V[v_00][col_offset + 1] += 1
        V[v_01][col_offset + 2] += 1
        V[v_11][col_offset + 2] += -1
        V[v_10][col_offset + 3] += 1
        V[v_11][col_offset + 4] += 1
        V[v_00][col_offset + 5] += 1
        V[v_01][col_offset + 5] += 1
        V[v_10][col_offset + 6] += 1
        V[v_11][col_offset + 6] += 1
        
        # W contributions
        # Z00 = M1 - M4 + M5 - M7
        W[w_00][col_offset + 0] += 1
        W[w_00][col_offset + 3] += -1
        W[w_00][col_offset + 4] += 1
        W[w_00][col_offset + 6] += -1
        
        # Z01 = M3 + M5
        W[w_01][col_offset + 2] += 1
        W[w_01][col_offset + 4] += 1
        
        # Z10 = M2 + M4
        W[w_10][col_offset + 1] += 1
        W[w_10][col_offset + 3] += 1
        
        # Z11 = M1 - M2 + M6
        W[w_11][col_offset + 0] += 1
        W[w_11][col_offset + 1] += -1
        W[w_11][col_offset + 5] += 1
        
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
