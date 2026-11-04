    def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
        """
        X_rows: [A00, A01, A10, A11]
        Y_rows: [B00, B01, B10, B11]
        Z_rows: [C00, C01, C10, C11] (indices in W)
        """
        a, b, c, d = X_rows
        e, f, g, h = Y_rows
        c00, c01, c10, c11 = Z_rows
        
        # M1 = (a+d)(e+h)
        # C00 += M1, C11 += M1
        U[a][col_offset + 0] += 1
        U[d][col_offset + 0] += 1
        V[e][col_offset + 0] += 1
        V[h][col_offset + 0] += 1
        W[c00][col_offset + 0] += 1
        W[c11][col_offset + 0] += 1

        # M2 = (c+d)e
        # C10 += M2, C11 -= M2
        U[c][col_offset + 1] += 1
        U[d][col_offset + 1] += 1
        V[e][col_offset + 1] += 1
        W[c10][col_offset + 1] += 1
        W[c11][col_offset + 1] -= 1

        # M3 = a(f-h)
        # C01 += M3, C11 += M3
        U[a][col_offset + 2] += 1
        V[f][col_offset + 2] += 1
        V[h][col_offset + 2] -= 1
        W[c01][col_offset + 2] += 1
        W[c11][col_offset + 2] += 1

        # M4 = d(g-e)
        # C00 += M4, C10 += M4
        U[d][col_offset + 3] += 1
        V[g][col_offset + 3] += 1
        V[e][col_offset + 3] -= 1
        W[c00][col_offset + 3] += 1
        W[c10][col_offset + 3] += 1

        # M5 = (a+b)h
        # C00 -= M5, C01 += M5
        U[a][col_offset + 4] += 1
        U[b][col_offset + 4] += 1
        V[h][col_offset + 4] += 1
        W[c00][col_offset + 4] -= 1
        W[c01][col_offset + 4] += 1

        # M6 = (c-a)(e+f)
        # C11 += M6
        U[c][col_offset + 5] += 1
        U[a][col_offset + 5] -= 1
        V[e][col_offset + 5] += 1
        V[f][col_offset + 5] += 1
        W[c11][col_offset + 5] += 1

        # M7 = (b-d)(g+h)
        # C00 += M7
        U[b][col_offset + 6] += 1
        U[d][col_offset + 6] -= 1
        V[g][col_offset + 6] += 1
        V[h][col_offset + 6] += 1
        W[c00][col_offset + 6] += 1
