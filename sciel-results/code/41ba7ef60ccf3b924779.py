import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # 8 Strassen blocks
    # A is 2x4, B is 4x8
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B is split into 4 2x2 blocks horizontally?
    # B_1 = B[0:2, 0:2], B_2 = B[0:2, 2:4], B_3 = B[0:2, 4:6], B_4 = B[0:2, 6:8]
    # B_5 = B[2:4, 0:2], B_6 = B[2:4, 2:4], B_7 = B[2:4, 4:6], B_8 = B[2:4, 6:8]
    
    blocks = [
        (0, 0, 0, 0), # A_L, B_1 -> C_1
        (2, 0, 0, 0), # A_R, B_5 -> C_1 (add)
        (0, 2, 0, 2), # A_L, B_2 -> C_2
        (2, 2, 0, 2), # A_R, B_6 -> C_2 (add)
        (0, 4, 0, 4), # A_L, B_3 -> C_3
        (2, 4, 0, 4), # A_R, B_7 -> C_3 (add)
        (0, 6, 0, 6), # A_L, B_4 -> C_4
        (2, 6, 0, 6), # A_R, B_8 -> C_4 (add)
    ]
    
    # Strassen indices for 2x2
    # U indices: 0,1,4,5 for A_L (0,0),(0,1),(1,0),(1,1)
    #            2,3,6,7 for A_R
    # V indices: 0,1,8,9 for B_1
    #            2,3,10,11 for B_2
    #            4,5,12,13 for B_3
    #            6,7,14,15 for B_4
    #            16,17,24,25 for B_5
    #            18,19,26,27 for B_6
    #            20,21,28,29 for B_7
    #            22,23,30,31 for B_8
    # W indices: 0,1,2,3 for C_1
    #            4,5,6,7 for C_2
    #            8,9,10,11 for C_3
    #            12,13,14,15 for C_4

    for (a_off, b_col_off, b_row_off, c_col_off) in blocks:
        # a_off: 0 for A_L, 2 for A_R
        # b_col_off: 0,2,4,6
        # b_row_off: 0,2 (but encoded in b_col_off? No, separate)
        # Let's use explicit indices
        
        # A indices
        a00 = 0*4 + a_off
        a01 = 0*4 + a_off + 1
        a10 = 1*4 + a_off
        a11 = 1*4 + a_off + 1
        
        # B indices
        # b_row_off is 0 or 2
        # b_col_off is 0, 2, 4, 6
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        # C indices
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0 # Wait, c_col_off is 0,2,4,6. C is 2x8.
        # C index is col*2 + row.
        # c00 is row 0, col c_col_off.
        # c10 is row 1, col c_col_off.
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        c01 = (c_col_off+1)*2 + 0
        
        # Strassen
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    assert col == 56
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
