import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for Y (2x2) and Z (2x2)
    # Y indices: 00, 01, 10, 11
    # Z indices: 00, 01, 10, 11
    # M_s = (sum coeff_y * Y) * (sum coeff_z * Z)
    
    # M_0: (Y00+Y11)(Z00+Z11)
    Y_0 = [1, 0, 0, 1]
    Z_0 = [1, 0, 0, 1]
    
    # M_1: (Y10+Y11)Z00
    Y_1 = [0, 0, 1, 1]
    Z_1 = [1, 0, 0, 0]
    
    # M_2: Y00(Z10-Z11)
    Y_2 = [1, 0, 0, 0]
    Z_2 = [0, 0, 1, -1]
    
    # M_3: Y11(Z01-Z00)
    Y_3 = [0, 0, 0, 1]
    Z_3 = [-1, 1, 0, 0]
    
    # M_4: (Y00+Y10)Z11
    Y_4 = [1, 0, 1, 0]
    Z_4 = [0, 0, 0, 1]
    
    # M_5: (Y11-Y00)(Z00+Z01)
    Y_5 = [0, 0, -1, 1]
    Z_5 = [1, 1, 0, 0]
    
    # M_6: (Y01-Y11)(Z10+Z11)
    Y_6 = [0, 1, 0, -1]
    Z_6 = [0, 0, 1, 1]
    
    Y_coeffs = [Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6]
    Z_coeffs = [Z_0, Z_1, Z_2, Z_3, Z_4, Z_5, Z_6]
    
    # Reconstruction coefficients for X_00, X_01, X_10, X_11
    # X_00 = M_0 + M_3 - M_4 - M_6
    # X_01 = M_1 + M_2
    # X_10 = M_5 + M_6
    # X_11 = M_0 - M_2 + M_3 + M_4
    # Map to list of 7 coeffs for each output element
    X_00_coeffs = [1, 0, 0, 1, -1, 0, -1]
    X_01_coeffs = [0, 1, 1, 0, 0, 0, 0]
    X_10_coeffs = [0, 0, 0, 0, 0, 1, 1]
    X_11_coeffs = [1, 0, -1, 1, 1, 0, 0]
    
    X_coeffs = [X_00_coeffs, X_01_coeffs, X_10_coeffs, X_11_coeffs]
    
    # Define the 4 instances
    # Each instance: (u_row_offsets, v_row_offsets, w_row_offsets, col_offset)
    # u_row_offsets: map local (a,b) to global (i,j)
    # v_row_offsets: map local (c,d) to global (j,k)
    # w_row_offsets: map local (e,f) to global (k,i)
    
    instances = []
    
    # Instance 1: A1 B11 -> C[:, 0:2]
    # Y=A1: i=a, j=b
    # Z=B11: j=c, k=d
    # X=P1: i=e, k=f
    instances.append({
        'y_map': lambda a, b: (a, b),
        'z_map': lambda c, d: (c, d),
        'x_map': lambda e, f: (f, e), # Returns (k, i) for W index
        'col_off': 0
    })
    
    # Instance 2: A2 B21 -> C[:, 0:2]
    # Y=A2: i=a, j=b+2
    # Z=B21: j=c+2, k=d
    # X=P2: i=e, k=f
    instances.append({
        'y_map': lambda a, b: (a, b+2),
        'z_map': lambda c, d: (c+2, d),
        'x_map': lambda e, f: (f, e),
        'col_off': 7
    })
    
    # Instance 3: A1 B12 -> C[:, 2:4]
    # Y=A1: i=a, j=b
    # Z=B12: j=c, k=d+2
    # X=P3: i=e, k=f+2
    instances.append({
        'y_map': lambda a, b: (a, b),
        'z_map': lambda c, d: (c, d+2),
        'x_map': lambda e, f: (f+2, e),
        'col_off': 14
    })
    
    # Instance 4: A2 B22 -> C[:, 2:4]
    # Y=A2: i=a, j=b+2
    # Z=B22: j=c+2, k=d+2
    # X=P4: i=e, k=f+2
    instances.append({
        'y_map': lambda a, b: (a, b+2),
        'z_map': lambda c, d: (c+2, d+2),
        'x_map': lambda e, f: (f+2, e),
        'col_off': 21
    })
    
    # Fill matrices
    for inst in instances:
        col_off = inst['col_off']
        y_map = inst['y_map']
        z_map = inst['z_map']
        x_map = inst['x_map']
        
        for s in range(7):
            r = col_off + s
            # Fill U and V
            for a in range(2):
                for b in range(2):
                    i, j = y_map(a, b)
                    U[i*m + j, r] = Y_coeffs[s][a*2 + b]
            
            for c in range(2):
                for d in range(2):
                    j, k = z_map(c, d)
                    V[j*p + k, r] = Z_coeffs[s][c*2 + d]
            
            # Fill W
            for e in range(2):
                for f in range(2):
                    k, i = x_map(e, f)
                    W[k*n + i, r] = X_coeffs[e*2 + f][s]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
