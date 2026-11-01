    def add_strassen_2x2x2(a_cols, b_rows, c_cols):
        nonlocal col
        # Local A indices: 0, 1
        # Local B indices: 0, 1
        # Local C indices: 0, 1
        
        # Term 1: (A00+A11)(B00+B11)
        U[0*m + a_cols[0]][col] = 1
        U[1*m + a_cols[1]][col] = 1
        V[b_rows[0]*p + c_cols[0]][col] = 1
        V[b_rows[1]*p + c_cols[1]][col] = 1
        W[c_cols[0]*n + 0][col] = 1
        W[c_cols[1]*n + 1][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[1*m + a_cols[0]][col] = 1
        U[1*m + a_cols[1]][col] = 1
        V[b_rows[0]*p + c_cols[0]][col] = 1
        W[c_cols[0]*n + 1][col] = 1
        col += 1
        
        # Term 3: A00(B01-B11)
        U[0*m + a_cols[0]][col] = 1
        V[b_rows[0]*p + c_cols[1]][col] = 1
        V[b_rows[1]*p + c_cols[1]][col] = -1
        W[c_cols[1]*n + 0][col] = 1
        col += 1
        
        # Term 4: A11(B10-B00)
        U[1*m + a_cols[1]][col] = 1
        V[b_rows[1]*p + c_cols[0]][col] = 1
        V[b_rows[0]*p + c_cols[0]][col] = -1
        W[c_cols[0]*n + 1][col] = 1
        W[c_cols[1]*n + 1][col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11
        U[0*m + a_cols[0]][col] = 1
        U[0*m + a_cols[1]][col] = 1
        V[b_rows[1]*p + c_cols[1]][col] = 1
        W[c_cols[0]*n + 0][col] = 1
        W[c_cols[1]*n + 0][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[1*m + a_cols[0]][col] = 1
        U[0*m + a_cols[0]][col] = -1
        V[b_rows[0]*p + c_cols[0]][col] = 1
        V[b_rows[0]*p + c_cols[1]][col] = 1
        W[c_cols[1]*n + 1][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[0*m + a_cols[1]][col] = 1
        U[1*m + a_cols[1]][col] = -1
        V[b_rows[1]*p + c_cols[0]][col] = 1
        V[b_rows[1]*p + c_cols[1]][col] = 1
        W[c_cols[0]*n + 0][col] = 1
        W[c_cols[1]*n + 1][col] = 1
        col += 1
