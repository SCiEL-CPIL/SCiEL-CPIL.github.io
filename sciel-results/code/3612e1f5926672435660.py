DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Contributions to C00, C01, C10, C11
    strassen_contrib = [
        (1, 0, 0, 1),    # M1
        (0, 0, 1, -1),   # M2
        (0, 1, 0, 1),    # M3
        (1, 0, 1, 0),    # M4
        (-1, 1, 0, 0),   # M5
        (0, 0, 0, 1),    # M6
        (1, 0, 0, 0),    # M7
    ]
    
    # A coefficients for each Strassen term
    # A indices: 00=0, 01=1, 10=2, 11=3
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1
        [(2, 1), (3, 1)],    # M2
        [(0, 1)],             # M3
        [(3, 1)],             # M4
        [(0, 1), (1, 1)],    # M5
        [(2, 1), (0, -1)],   # M6
        [(1, 1), (3, -1)],   # M7
    ]
    
    # B block coefficients for each Strassen term
    # B indices: row 0 or 1, block 0 (cols 0-4) or 1 (cols 5-9)
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(0, 0, 1)],               # M2
        [(0, 1, 1), (1, 1, -1)],  # M3
        [(1, 0, 1), (0, 0, -1)],  # M4
        [(1, 1, 1)],               # M5
        [(0, 0, 1), (0, 1, 1)],   # M6
        [(1, 0, 1), (1, 1, 1)],   # M7
    ]
    
    col = 0
    for s in range(7):
        a_coeff_list = A_coeffs[s]
        b_coeff_list = B_coeffs[s]
        contrib = strassen_contrib[s]
        
        for t in range(5):
            # U: A entries
            for a_idx, a_c in a_coeff_list:
                U[a_idx][col] = a_c
            
            # V: B entries
            for b_row, b_block, b_c in b_coeff_list:
                b_col = b_block * 5 + t
                V[b_row * p + b_col][col] = b_c
            
            # W: C entries
            c00_c, c01_c, c10_c, c11_c = contrib
            if c00_c != 0:
                W[t * n + 0][col] = c00_c
            if c01_c != 0:
                W[(5 + t) * n + 0][col] = c01_c
            if c10_c != 0:
                W[t * n + 1][col] = c10_c
            if c11_c != 0:
                W[(5 + t) * n + 1][col] = c11_c
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
