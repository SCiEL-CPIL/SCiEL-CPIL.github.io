import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    # 8 blocks of 2x2x2 Strassen? No, 4 blocks of 2x4x4?
    # The previous logic used 8 blocks of 2x2x2?
    # Let's re-verify the block structure in the current code.
    # blocks = [
    #     (0, 0, 0, 0), # a_col=0, b_row=0, b_col=0, c_col=0
    #     (0, 0, 2, 2),
    #     (0, 0, 4, 4),
    #     (0, 0, 6, 6),
    #     (2, 2, 0, 0),
    #     (2, 2, 2, 2),
    #     (2, 2, 4, 4),
    #     (2, 2, 6, 6),
    # ]
    # This iterates 8 times.
    # Inside, it adds 7 terms.
    # Total 56 terms.
    # This corresponds to 8 independent 2x2x2 Strassen multiplications?
    # A is 2x4. B is 4x8.
    # Block (a_col_off, b_row_off, b_col_off, c_col_off)
    # A block is 2x2 at cols a_col_off, a_col_off+1.
    # B block is 2x2 at rows b_row_off, b_row_off+1 and cols b_col_off, b_col_off+1.
    # C block is 2x2 at cols c_col_off, c_col_off+1?
    # Wait, C is 2x8.
    # If we use 2x2x2 Strassen, the output is 2x2.
    # The blocks cover:
    # A cols: 0-1, 2-3.
    # B rows: 0-1, 2-3.
    # B cols: 0-1, 2-3, 4-5, 6-7.
    # This covers all of A and B.
    # And C cols: 0-1, 2-3, 4-5, 6-7.
    # This covers all of C.
    # So it is 8 independent 2x2x2 multiplications.
    # Rank is 8 * 7 = 56.
    # This is a valid decomposition.
    
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        # Strassen 7 terms
        # 1
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # 2
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # 3
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # 4
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # 5
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # 6
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # 7
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
