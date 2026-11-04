def add_strassen_block(U, V, W, i_start, j_start, k_start, col_offset):
    # i_start, j_start, k_start are the starting indices of the 2x2 block
    # A is rows i_start..i_start+1, cols j_start..j_start+1
    # B is rows j_start..j_start+1, cols k_start..k_start+1
    # C is rows i_start..i_start+1, cols k_start..k_start+1
    
    # Strassen coefficients for A (2x2)
    # M1: (a+d)(e+h) -> A: a, d
    # M2: (c+d)e    -> A: c, d
    # M3: a(f-h)    -> A: a
    # M4: d(g-e)    -> A: d
    # M5: (a+b)h    -> A: a, b
    # M6: (c-a)(e+f) -> A: c, -a
    # M7: (b-d)(g+h) -> A: b, -d
    
    a_coeffs = [
        [1, 0], # M1: a
        [0, 1], # M2: d (wait, M2 is c+d, so A has c, d. c is index 1,0? No. A is [[a,b],[c,d]]. c is (1,0), d is (1,1).)
        # Let's list A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # M1: a, d
        # M2: c, d
        # M3: a
        # M4: d
        # M5: a, b
        # M6: c, -a
        # M7: b, -d
    ]
    
    # Correct A coefficients for each term (2x2 matrix)
    A_coeffs = [
        [[1, 0], [0, 1]], # M1: a, d
        [[0, 0], [1, 1]], # M2: c, d
        [[1, 0], [0, 0]], # M3: a
        [[0, 0], [0, 1]], # M4: d
        [[1, 1], [0, 0]], # M5: a, b
        [[-1, 0], [1, 0]], # M6: -a, c
        [[0, 1], [0, -1]]  # M7: b, -d
    ]
    
    # B coefficients for each term (2x2 matrix)
    # M1: (a+d)(e+h) -> B: e, h
    # M2: (c+d)e    -> B: e
    # M3: a(f-h)    -> B: f, -h
    # M4: d(g-e)    -> B: g, -e
    # M5: (a+b)h    -> B: h
    # M6: (c-a)(e+f) -> B: e, f
    # M7: (b-d)(g+h) -> B: g, h
    B_coeffs = [
        [[1, 0], [0, 1]], # M1: e, h
        [[1, 0], [0, 0]], # M2: e
        [[0, 1], [0, -1]], # M3: f, -h
        [[0, -1], [1, 0]], # M4: -e, g
        [[0, 0], [0, 1]], # M5: h
        [[1, 1], [0, 0]], # M6: e, f
        [[0, 0], [1, 1]]  # M7: g, h
    ]
    
    # C coefficients for each term (2x2 matrix)
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    C_coeffs = [
        [[1, 0], [0, 0]], # M1: C00, C11
        [[0, 0], [-1, 0]], # M2: -C10? No. C11 has -M2. C10 has M2.
        # Let's re-verify C formula
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00=1, C11=1
        # M2: C10=1, C11=-1
        # M3: C01=1, C11=1
        # M4: C00=1, C10=1
        # M5: C00=-1, C01=1
        # M6: C11=1
        # M7: C00=1
    ]
    
    # Correct C coefficients:
    C_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    for t in range(7):
        col = col_offset + t
        for di in range(2):
            for dj in range(2):
                val = A_coeffs[t][di][dj]
                if val != 0:
                    U[(i_start + di) * 5 + (j_start + dj)][col] += val
        for dj in range(2):
            for dk in range(2):
                val = B_coeffs[t][dj][dk]
                if val != 0:
                    V[(j_start + dj) * 6 + (k_start + dk)][col] += val
        for di in range(2):
            for dk in range(2):
                val = C_coeffs[t][di][dk]
                if val != 0:
                    W[(k_start + dk) * 2 + (i_start + di)][col] += val

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (2x1) * B1 (1x6)
    # A1 is col 0. B1 is row 0.
    # This is 12 terms.
    for i in range(2):
        for k in range(6):
            U[i * 5 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Block 2: A2 (2x4) * B2 (4x6)
    # A2 cols 1-4. B2 rows 1-4.
    # Split B2 into B2a (4x4, cols 0-3) and B2b (4x2, cols 4-5)
    
    # Sub-block 2a: A2 (2x4) * B2a (4x4)
    # Split A2 into A2a (2x2, cols 1-2) and A2b (2x2, cols 3-4)
    # Split B2a into B2a1 (2x4, rows 1-2) and B2a2 (2x4, rows 3-4)
    
    # Term A2a * B2a1
    # A2a: rows 0-1, cols 1-2
    # B2a1: rows 1-2, cols 0-3
    # Split B2a1 into B2a1a (2x2, cols 0-1) and B2a1b (2x2, cols 2-3)
    
    # A2a * B2a1a
    add_strassen_block(U, V, W, 0, 1, 0, col); col += 7
    # A2a * B2a1b
    add_strassen_block(U, V, W, 0, 1, 2, col); col += 7
    
    # Term A2b * B2a2
    # A2b: rows 0-1, cols 3-4
    # B2a2: rows 3-4, cols 0-3
    # Split B2a2 into B2a2a (2x2, cols 0-1) and B2a2b (2x2, cols 2-3)
    
    # A2b * B2a2a
    add_strassen_block(U, V, W, 0, 3, 0, col); col += 7
    # A2b * B2a2b
    add_strassen_block(U, V, W, 0, 3, 2, col); col += 7
    
    # Sub-block 2b: A2 (2x4) * B2b (4x2)
    # B2b: rows 1-4, cols 4-5
    # Split A2 into A2c (2x2, cols 1-2) and A2d (2x2, cols 3-4)
    # Split B2b into B2b1 (2x2, rows 1-2) and B2b2 (2x2, rows 3-4)
    
    # A2c * B2b1
    add_strassen_block(U, V, W, 0, 1, 4, col); col += 7
    # A2d * B2b2
    add_strassen_block(U, V, W, 0, 3, 4, col); col += 7
    
    return {"U": U, "V": V, "W": W}
