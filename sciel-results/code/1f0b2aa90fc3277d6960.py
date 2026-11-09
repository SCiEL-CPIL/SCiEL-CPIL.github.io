DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen coefficients for 2x2x2
    strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    sub_products = [
        (0, 0, 0),
        (1, 2, 0),
        (0, 1, 1),
        (1, 3, 1),
    ]
    
    for a_block, b_block, c_block in sub_products:
        for x_coeffs, y_coeffs, z_coeffs in strassen:
            for (xi, xj), xc in x_coeffs.items():
                a_row = xi
                a_col = (2 if a_block == 1 else 0) + xj
                U[a_row * m + a_col][col] += xc
            
            for (yi, yj), yc in y_coeffs.items():
                b_row = (2 if b_block >= 2 else 0) + yi
                b_col = (2 if b_block % 2 == 1 else 0) + yj
                V[b_row * p + b_col][col] += yc
            
            for (zi, zj), zc in z_coeffs.items():
                c_row = zi
                c_col = (2 if c_block == 1 else 0) + zj
                W[c_col * n + c_row][col] += zc
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
