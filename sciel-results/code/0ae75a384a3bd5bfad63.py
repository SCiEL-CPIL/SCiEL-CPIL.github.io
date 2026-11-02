import time
DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, a_coeffs, b_coeffs, c_coeffs):
        for (i, j), val in a_coeffs.items():
            U[i * m + j][col] += val
        for (j, k), val in b_coeffs.items():
            V[j * p + k][col] += val
        for (i, k), val in c_coeffs.items():
            W[k * n + i][col] += val

    def add_strassen(start_col, a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x2 at (a_r, a_c)
        # B is 2x2 at (b_r, b_c)
        # C is 2x2 at (c_r, c_c)
        
        # m1 = (A00+A11)(B00+B11)
        add_term(start_col, 
                 {(a_r, a_c): 1, (a_r+1, a_c+1): 1}, 
                 {(b_r, b_c): 1, (b_r+1, b_c+1): 1}, 
                 {(c_r, c_c): 1, (c_r+1, c_c+1): 1})
        
        # m2 = (A10+A11)B00
        add_term(start_col+1, 
                 {(a_r+1, a_c): 1, (a_r+1, a_c+1): 1}, 
                 {(b_r, b_c): 1}, 
                 {(c_r+1, c_c): 1})
                 
        # m3 = A00(B10-B11)
        add_term(start_col+2, 
                 {(a_r, a_c): 1}, 
                 {(b_r+1, b_c): 1, (b_r+1, b_c+1): -1}, 
                 {(c_r, c_c+1): 1, (c_r+1, c_c): 1})
                 
        # m4 = A11(B01+B11)
        add_term(start_col+3, 
                 {(a_r+1, a_c+1): 1}, 
                 {(b_r, b_c+1): 1, (b_r+1, b_c+1): 1}, 
                 {(c_r, c_c): -1, (c_r+1, c_c+1): 1})
                 
        # m5 = (A00+A01)B11
        add_term(start_col+4, 
                 {(a_r, a_c): 1, (a_r, a_c+1): 1}, 
                 {(b_r+1, b_c+1): 1}, 
                 {(c_r, c_c): 1, (c_r+1, c_c+1): -1})
                 
        # m6 = (A10-A00)(B00+B01)
        add_term(start_col+5, 
                 {(a_r+1, a_c): 1, (a_r, a_c): -1}, 
                 {(b_r, b_c): 1, (b_r, b_c+1): 1}, 
                 {(c_r+1, c_c): 1, (c_r+1, c_c+1): 1})
                 
        # m7 = (A01-A11)(B10+B11)
        add_term(start_col+6, 
                 {(a_r, a_c+1): 1, (a_r+1, a_c+1): -1}, 
                 {(b_r+1, b_c): 1, (b_r+1, b_c+1): 1}, 
                 {(c_r, c_c+1): -1, (c_r+1, c_c+1): -1})

    def add_schoolbook_2x2x2x1(start_col, a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x2 at (a_r, a_c)
        # B is 2x1 at (b_r, b_c)
        # C is 2x1 at (c_r, c_c)
        
        # C00 = A00 B00 + A01 B10
        add_term(start_col, 
                 {(a_r, a_c): 1}, 
                 {(b_r, b_c): 1}, 
                 {(c_r, c_c): 1})
                 
        add_term(start_col+1, 
                 {(a_r, a_c+1): 1}, 
                 {(b_r+1, b_c): 1}, 
                 {(c_r, c_c): 1})
                 
        # C10 = A10 B00 + A11 B10
        add_term(start_col+2, 
                 {(a_r+1, a_c): 1}, 
                 {(b_r, b_c): 1}, 
                 {(c_r+1, c_c): 1})
                 
        add_term(start_col+3, 
                 {(a_r+1, a_c+1): 1}, 
                 {(b_r+1, b_c): 1}, 
                 {(c_r+1, c_c): 1})

    col = 0
    # Group 1: A1 (rows 0-1, cols 0-1) * B1a (rows 0-1, cols 0-1) -> C (rows 0-1, cols 0-1)
    add_strassen(col, 0, 0, 0, 0, 0, 0)
    col += 7
    
    # Group 2: A1 * B1b1 (rows 0-1, cols 2-3) -> C (rows 0-1, cols 2-3)
    add_strassen(col, 0, 0, 0, 2, 0, 2)
    col += 7
    
    # Group 3: A1 * B1b2 (rows 0-1, col 4) -> C (rows 0-1, col 4)
    add_schoolbook_2x2x2x1(col, 0, 0, 0, 4, 0, 4)
    col += 4
    
    # Group 4: A2 (rows 0-1, cols 2-3) * B2a (rows 2-3, cols 0-1) -> C (rows 0-1, cols 0-1)
    add_strassen(col, 0, 2, 2, 0, 0, 0)
    col += 7
    
    # Group 5: A2 * B2b1 (rows 2-3, cols 2-3) -> C (rows 0-1, cols 2-3)
    add_strassen(col, 0, 2, 2, 2, 0, 2)
    col += 7
    
    # Group 6: A2 * B2b2 (rows 2-3, col 4) -> C (rows 0-1, col 4)
    add_schoolbook_2x2x2x1(col, 0, 2, 2, 4, 0, 4)
    col += 4
    
    return {"U": U, "V": V, "W": W}
