import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for X, Y, Z
    # M1 = (X11+X22)(Y11+Y22) -> Z11+1, Z22+1
    # M2 = (X21+X22)Y11 -> Z21+1, Z22-1
    # M3 = X11(Y12-Y22) -> Z12+1, Z22+1
    # M4 = X22(Y21-Y11) -> Z11+1, Z21+1
    # M5 = (X11+X12)Y22 -> Z11-1, Z12+1
    # M6 = (X21-X11)(Y11+Y12) -> Z22+1
    # M7 = (X12-X22)(Y21+Y22) -> Z11+1
    
    # Format: (term_id, x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict { (row, col): coeff }
    # y_coeffs: dict { (row, col): coeff }
    # z_coeffs: dict { (row, col): coeff }
    
    terms = [
        (1, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        (2, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        (3, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        (4, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        (5, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        (6, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        (7, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    # Loop over 4 column blocks
    for L in range(4):
        b_start_col = 2 * L
        
        # Top block: A_left * B_T_L
        a_start_col = 0
        b_start_row = 0
        
        for t_id, x_c, y_c, z_c in terms:
            # Update U
            for (xr, xc), val in x_c.items():
                u_idx = xr * m + (a_start_col + xc)
                U[u_idx, col] += val
            
            # Update V
            for (yr, yc), val in y_c.items():
                v_idx = (b_start_row + yr) * p + (b_start_col + yc)
                V[v_idx, col] += val
            
            # Update W
            for (zr, zc), val in z_c.items():
                # Z is 2x2 block at C[0:2, 2L:2L+2]
                # zr is row in Z (0,1), zc is col in Z (0,1)
                c_row = zr
                c_col = b_start_col + zc
                w_idx = c_col * n + c_row
                W[w_idx, col] += val
            
            col += 1
        
        # Bottom block: A_right * B_B_L
        a_start_col = 2
        b_start_row = 2
        
        for t_id, x_c, y_c, z_c in terms:
            # Update U
            for (xr, xc), val in x_c.items():
                u_idx = xr * m + (a_start_col + xc)
                U[u_idx, col] += val
            
            # Update V
            for (yr, yc), val in y_c.items():
                v_idx = (b_start_row + yr) * p + (b_start_col + yc)
                V[v_idx, col] += val
            
            # Update W
            for (zr, zc), val in z_c.items():
                c_row = zr
                c_col = b_start_col + zc
                w_idx = c_col * n + c_row
                W[w_idx, col] += val
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
