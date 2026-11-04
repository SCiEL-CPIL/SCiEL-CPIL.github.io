def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_col_off, b_row_off, b_col_off):
        nonlocal col
        
        # Term 1: (A00 + A11)(B00 + B11)
        U[0*2 + a_col_off + 0][col] = 1
        U[1*2 + a_col_off + 1][col] = 1
        V[(b_row_off + 0)*6 + (b_col_off + 0)][col] = 1
        V[(b_row_off + 1)*6 + (b_col_off + 1)][col] = 1
        W[0*2 + 0][col] = 1
        W[1*2 + 1][col] = 1
        col += 1
        
        # Term 2: (A10 + A11)B00
        U[1*2 + a_col_off + 0][col] = 1
        U[1*2 + a_col_off + 1][col] = 1
        V[(b_row_off + 0)*6 + (b_col_off + 0)][col] = 1
        W[0*2 + 1][col] = 1
        W[1*2 + 1][col] = -1
        col += 1
        
        # Term 3: A00(B10 - B11)
        U[0*2 + a_col_off + 0][col] = 1
        V[(b_row_off + 1)*6 + (b_col_off + 0)][col] = 1
        V[(b_row_off + 1)*6 + (b_col_off + 1)][col] = -1
        W[1*2 + 0][col] = 1
        W[1*2 + 1][col] = 1
        col += 1
        
        # Term 4: A11(B01 - B00)
        U[1*2 + a_col_off + 1][col] = 1
        V[(b_row_off + 0)*6 + (b_col_off + 1)][col] = 1
        V[(b_row_off + 0)*6 + (b_col_off + 0)][col] = -1
        W[0*2 + 0][col] = 1
        W[0*2 + 1][col] = 1
        col += 1
        
        # Term 5: (A00 + A01)B11
        U[0*2 + a_col_off + 0][col] = 1
        U[0*2 + a_col_off + 1][col] = 1
        V[(b_row_off + 1)*6 + (b_col_off + 1)][col] = 1
        W[0*2 + 0][col] = -1
        W[1*2 + 0][col] = 1
        col += 1
        
        # Term 6: (A10 - A00)(B00 + B01)
        U[1*2 + a_col_off + 0][col] = 1
        U[0*2 + a_col_off + 0][col] = -1
        V[(b_row_off + 0)*6 + (b_col_off + 0)][col] = 1
        V[(b_row_off + 0)*6 + (b_col_off + 1)][col] = 1
        W[1*2 + 1][col] = 1
        col += 1
        
        # Term 7: (A01 - A11)(B10 + B11)
        U[0*2 + a_col_off + 1][col] = 1
        U[1*2 + a_col_off + 1][col] = -1
        V[(b_row_off + 1)*6 + (b_col_off + 0)][col] = 1
        V[(b_row_off + 1)*6 + (b_col_off + 1)][col] = 1
        W[0*2 + 0][col] = 1
        col += 1

    # Group 1
    for x in range(3):
        add_strassen(0, 0, 2*x)
        
    # Group 2
    for x in range(3):
        add_strassen(2, 2, 2*x)
        
    # Group 3
    for i in range(2):
        for k in range(6):
            U[i*5 + 4][col] = 1
            V[4*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
