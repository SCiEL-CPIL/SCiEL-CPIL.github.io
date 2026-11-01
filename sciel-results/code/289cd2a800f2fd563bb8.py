    def add_block(U, V, W, col_offset, a_row, a_col, b_row, b_col, c_row, c_col, signs):
        # a_row, a_col: top-left of A block (2x2)
        # b_row, b_col: top-left of B block (2x2)
        # c_row, c_col: top-left of C block (2x2)
        
        # Indices:
        # U: i*4 + j
        # V: j*4 + k
        # W: k*2 + i  (since n=2)
        
        # M1 = (A11+A22)(B11+B22)
        # U: (a_row, a_col) +1, (a_row+1, a_col+1) +1
        # V: (b_row, b_col) +1, (b_row+1, b_col+1) +1
        # W: C11 (c_row, c_col) +1, C22 (c_row+1, c_col+1) +1
        U[a_row*4+a_col, col_offset+0] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+0] += 1
        V[b_row*4+b_col, col_offset+0] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+0] += 1
        W[c_col*2+c_row, col_offset+0] += 1
        W[(c_col+1)*2+(c_row+1), col_offset+0] += 1
        
        # M2 = (A21+A22)B11
        # U: (a_row+1, a_col) +1, (a_row+1, a_col+1) +1
        # V: (b_row, b_col) +1
        # W: C21 (c_row+1, c_col) +1
        U[(a_row+1)*4+a_col, col_offset+1] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+1] += 1
        V[b_row*4+b_col, col_offset+1] += 1
        W[c_col*2+(c_row+1), col_offset+1] += 1
        
        # M3 = A11(B12-B22)
        # U: (a_row, a_col) +1
        # V: (b_row, b_col+1) +1, (b_row+1, b_col+1) -1
        # W: C12 (c_row, c_col+1) +1, C22 (c_row+1, c_col+1) +1
        U[a_row*4+a_col, col_offset+2] += 1
        V[b_row*4+(b_col+1), col_offset+2] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+2] -= 1
        W[(c_col+1)*2+c_row, col_offset+2] += 1
        W[(c_col+1)*2+(c_row+1), col_offset+2] += 1
        
        # M4 = A22(B21-B11)
        # U: (a_row+1, a_col+1) +1
        # V: (b_row+1, b_col) +1, (b_row, b_col) -1
        # W: C11 (c_row, c_col) +1, C21 (c_row+1, c_col) +1
        U[(a_row+1)*4+(a_col+1), col_offset+3] += 1
        V[(b_row+1)*4+b_col, col_offset+3] += 1
        V[b_row*4+b_col, col_offset+3] -= 1
        W[c_col*2+c_row, col_offset+3] += 1
        W[c_col*2+(c_row+1), col_offset+3] += 1
        
        # M5 = (A11+A12)B22
        # U: (a_row, a_col) +1, (a_row, a_col+1) +1
        # V: (b_row+1, b_col+1) +1
        # W: C11 (c_row, c_col) -1, C12 (c_row, c_col+1) +1
        U[a_row*4+a_col, col_offset+4] += 1
        U[a_row*4+(a_col+1), col_offset+4] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+4] += 1
        W[c_col*2+c_row, col_offset+4] -= 1
        W[(c_col+1)*2+c_row, col_offset+4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        # U: (a_row+1, a_col) +1, (a_row, a_col) -1
        # V: (b_row, b_col) +1, (b_row, b_col+1) +1
        # W: C22 (c_row+1, c_col+1) +1
        U[(a_row+1)*4+a_col, col_offset+5] += 1
        U[a_row*4+a_col, col_offset+5] -= 1
        V[b_row*4+b_col, col_offset+5] += 1
        V[b_row*4+(b_col+1), col_offset+5] += 1
        W[(c_col+1)*2+(c_row+1), col_offset+5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        # U: (a_row, a_col+1) +1, (a_row+1, a_col+1) -1
        # V: (b_row+1, b_col) +1, (b_row+1, b_col+1) +1
        # W: C11 (c_row, c_col) +1, C22 (c_row+1, c_col+1) +1
        U[a_row*4+(a_col+1), col_offset+6] += 1
        U[(a_row+1)*4+(a_col+1), col_offset+6] -= 1
        V[(b_row+1)*4+b_col, col_offset+6] += 1
        V[(b_row+1)*4+(b_col+1), col_offset+6] += 1
        W[c_col*2+c_row, col_offset+6] += 1
        W[(c_col+1)*2+(c_row+1), col_offset+6] += 1
