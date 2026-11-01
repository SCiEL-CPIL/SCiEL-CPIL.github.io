    def add_strassen(b_00, b_01, b_10, b_11, col_offset):
        # P1 = A00 * (B00 + B11)
        U[0][col_offset + 0] = 1
        V[b_00][col_offset + 0] = 1
        V[b_11][col_offset + 0] = 1
        W[0][col_offset + 0] = 1
        W[3][col_offset + 0] = 1
        
        # P2 = (A10 + A11) * B00
        U[2][col_offset + 1] = 1
        U[3][col_offset + 1] = 1
        V[b_00][col_offset + 1] = 1
        W[2][col_offset + 1] = 1
        
        # P3 = (A00 + A01) * B11
        U[0][col_offset + 2] = 1
        U[1][col_offset + 2] = 1
        V[b_11][col_offset + 2] = 1
        W[1][col_offset + 2] = 1
        W[3][col_offset + 2] = 1
        
        # P4 = A11 * (B01 - B00)
        U[3][col_offset + 3] = 1
        V[b_01][col_offset + 3] = 1
        V[b_00][col_offset + 3] = -1
        W[0][col_offset + 3] = 1
        
        # P5 = A00 * (B10 - B11)
        U[0][col_offset + 4] = 1
        V[b_10][col_offset + 4] = 1
        V[b_11][col_offset + 4] = -1
        W[0][col_offset + 4] = -1
        W[2][col_offset + 4] = 1
        
        # P6 = (A10 - A00) * B00
        U[2][col_offset + 5] = 1
        U[0][col_offset + 5] = -1
        V[b_00][col_offset + 5] = 1
        W[3][col_offset + 5] = 1
        
        # P7 = (A01 - A11) * B11
        U[1][col_offset + 6] = 1
        U[3][col_offset + 6] = -1
        V[b_11][col_offset + 6] = 1
        W[0][col_offset + 6] = -1
        W[1][col_offset + 6] = 1
