import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x2
        # M1 = (A00+A11)(B00+B11)
        U[a_row_offset*4 + a_col_offset][col] = 1
        U[(a_row_offset+1)*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset)*7 + b_col_offset][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+1)][col] = 1
        # C00 += M1, C11 += M1
        W[(c_col_offset)*2 + c_row_offset][col] = 1
        W[(c_col_offset+1)*2 + (c_row_offset+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(a_row_offset+1)*4 + a_col_offset][col] = 1
        U[(a_row_offset+1)*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset)*7 + b_col_offset][col] = 1
        # C10 += M2, C11 -= M2
        W[(c_col_offset)*2 + (c_row_offset+1)][col] = 1
        W[(c_col_offset+1)*2 + (c_row_offset+1)][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_row_offset*4 + a_col_offset][col] = 1
        V[(b_row_offset)*7 + (b_col_offset+1)][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+1)][col] = -1
        # C01 += M3, C11 += M3
        W[(c_col_offset+1)*2 + c_row_offset][col] = 1
        W[(c_col_offset+1)*2 + (c_row_offset+1)][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(a_row_offset+1)*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1)*7 + b_col_offset][col] = 1
        V[(b_row_offset)*7 + b_col_offset][col] = -1
        # C00 += M4, C10 += M4
        W[(c_col_offset)*2 + c_row_offset][col] = 1
        W[(c_col_offset)*2 + (c_row_offset+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_row_offset*4 + a_col_offset][col] = 1
        U[a_row_offset*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+1)][col] = 1
        # C00 -= M5, C01 += M5
        W[(c_col_offset)*2 + c_row_offset][col] = -1
        W[(c_col_offset+1)*2 + c_row_offset][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(a_row_offset+1)*4 + a_col_offset][col] = 1
        U[a_row_offset*4 + a_col_offset][col] = -1
        V[(b_row_offset)*7 + b_col_offset][col] = 1
        V[(b_row_offset)*7 + (b_col_offset+1)][col] = 1
        # C11 += M6
        W[(c_col_offset+1)*2 + (c_row_offset+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_row_offset*4 + (a_col_offset+1)][col] = 1
        U[(a_row_offset+1)*4 + (a_col_offset+1)][col] = -1
        V[(b_row_offset+1)*7 + b_col_offset][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+1)][col] = 1
        # C00 += M7
        W[(c_col_offset)*2 + c_row_offset][col] = 1
        col += 1

    def add_223_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A is 2x2, B is 2x3
        # First 2 cols of B: Strassen
        add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        # Last col of B: Schoolbook
        # M1 = A00 B02
        U[a_row_offset*4 + a_col_offset][col] = 1
        V[(b_row_offset)*7 + (b_col_offset+2)][col] = 1
        W[(c_col_offset+2)*2 + c_row_offset][col] = 1
        col += 1
        
        # M2 = A01 B12
        U[a_row_offset*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+2)][col] = 1
        W[(c_col_offset+2)*2 + c_row_offset][col] = 1
        col += 1
        
        # M3 = A10 B02
        U[(a_row_offset+1)*4 + a_col_offset][col] = 1
        V[(b_row_offset)*7 + (b_col_offset+2)][col] = 1
        W[(c_col_offset+2)*2 + (c_row_offset+1)][col] = 1
        col += 1
        
        # M4 = A11 B12
        U[(a_row_offset+1)*4 + (a_col_offset+1)][col] = 1
        V[(b_row_offset+1)*7 + (b_col_offset+2)][col] = 1
        W[(c_col_offset+2)*2 + (c_row_offset+1)][col] = 1
        col += 1

    # Block 1: C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 2: C[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 3: C[0:2, 4:7]
    # A[0:2, 0:2] * B[0:2, 4:7]
    add_223_block(0, 0, 0, 4, 0, 4)
    # A[0:2, 2:4] * B[2:4, 4:7]
    add_223_block(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
