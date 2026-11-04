DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Index U[i*m + j], V[j*p + k], W[k*n + i]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen products for a 2x2 sub-multiplication
    # Each product: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (di,dj) to coeff
    # b_coeffs: dict from (dj,dk) to coeff
    # c_coeffs: dict from (di,dk) to coeff
    
    strassen_products = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 4 sub-multiplications: (i0, j0, k0)
    # A0*B00: i0=0, j0=0, k0=0
    # A1*B10: i0=0, j0=2, k0=0
    # A0*B01: i0=0, j0=0, k0=2
    # A1*B11: i0=0, j0=2, k0=2
    # Wait, A is 2x4, so A0 is cols 0,1 and A1 is cols 2,3.
    # B is 4x4, B00 is rows 0,1 cols 0,1; B01 is rows 0,1 cols 2,3; B10 is rows 2,3 cols 0,1; B11 is rows 2,3 cols 2,3.
    # C is 2x4, C0 is cols 0,1; C1 is cols 2,3.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    sub_mults = [
        (0, 0, 0),  # A0*B00 -> C00
        (0, 2, 0),  # A1*B10 -> C00 (adds to C0)
        (0, 0, 2),  # A0*B01 -> C01
        (0, 2, 2),  # A1*B11 -> C01
    ]
    
    for (i0, j0, k0) in sub_mults:
        for (a_coeffs, b_coeffs, c_coeffs) in strassen_products:
            for (di, dj), val in a_coeffs.items():
                U[(i0+di)*m + (j0+dj)][col] += val
            for (dj, dk), val in b_coeffs.items():
                V[(j0+dj)*p + (k0+dk)][col] += val
            for (di, dk), val in c_coeffs.items():
                W[(k0+dk)*n + (i0+di)][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
