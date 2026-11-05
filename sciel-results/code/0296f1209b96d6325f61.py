import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a10 + a11) b00
    # M2 = (a00 + a11) b11
    # M3 = a00 (b01 - b11)
    # M4 = a11 (b10 - b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 - a00) (b00 + b01)
    # M7 = (a01 - a11) (b10 + b11)
    
    # C00 = M1 + M2 - M3 + M7
    # C01 = M3 + M5
    # C10 = M4 + M6
    # C11 = M1 - M2 + M4 + M5
    
    strassen_A = [
        {1: 1, 3: 1},      # M1: a10, a11 (local indices 1,0 and 1,1? No. a10 is row 1 col 0)
        # Local A indices: (0,0), (0,1), (1,0), (1,1)
        # M1: (1,0), (1,1)
        # M2: (0,0), (1,1)
        # M3: (0,0)
        # M4: (1,1)
        # M5: (0,0), (0,1)
        # M6: (1,0), (0,0)
        # M7: (0,1), (1,1)
    ]
    # Let's just hardcode the contributions for clarity and correctness.
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, start_col):
        # a_row_offset, a_col_offset: top-left of A block (2x2)
        # b_row_offset, b_col_offset: top-left of B block (2x2)
        # b_col_offset is also C col offset
        
        # M1
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][start_col] = 1
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][start_col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][start_col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 0)][start_col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 1)][start_col] = 1
        col = start_col + 1
        
        # M2
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] = 1
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 0)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 1)][col] = -1
        col += 1
        
        # M3
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] = -1
        W[(b_col_offset + 0) * n + (a_row_offset + 0)][col] = -1
        W[(b_col_offset + 1) * n + (a_row_offset + 0)][col] = 1
        col += 1
        
        # M4
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * p + (b_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] = -1
        W[(b_col_offset + 0) * n + (a_row_offset + 1)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 1)][col] = 1
        col += 1
        
        # M5
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] = 1
        U[(a_row_offset + 0) * m + (a_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 0)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 1)][col] = 1
        col += 1
        
        # M6
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][col] = 1
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] = -1
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 1)][col] = 1
        col += 1
        
        # M7
        U[(a_row_offset + 0) * m + (a_col_offset + 1)][col] = 1
        U[(a_row_offset + 1) * m + (a_col_offset + 1)][col] = -1
        V[(b_row_offset + 1) * p + (b_col_offset + 0)][col] = 1
        V[(b_row_offset + 1) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 0)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 0)][col] = 1
        col += 1
        
        return col

    def add_outer_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, start_col):
        # A block is 2x1. B block is 1x2.
        # a_row_offset, a_col_offset: top-left of A block
        # b_row_offset, b_col_offset: top-left of B block
        
        # Term 1: a00 b00 -> C00
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][start_col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][start_col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 0)][start_col] = 1
        col = start_col + 1
        
        # Term 2: a00 b01 -> C01
        U[(a_row_offset + 0) * m + (a_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 0)][col] = 1
        col += 1
        
        # Term 3: a10 b00 -> C10
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 0)][col] = 1
        W[(b_col_offset + 0) * n + (a_row_offset + 1)][col] = 1
        col += 1
        
        # Term 4: a10 b01 -> C11
        U[(a_row_offset + 1) * m + (a_col_offset + 0)][col] = 1
        V[(b_row_offset + 0) * p + (b_col_offset + 1)][col] = 1
        W[(b_col_offset + 1) * n + (a_row_offset + 1)][col] = 1
        col += 1
        
        return col

    # A is 2x5. Split cols into 2, 2, 1.
    # B is 5x6. Split rows into 2, 2, 1. Split cols into 2, 2, 2.
    
    a_col_splits = [0, 2, 4] # start indices for A cols
    b_row_splits = [0, 2, 4] # start indices for B rows
    b_col_splits = [0, 2, 4] # start indices for B cols
    
    for b_c in range(3):
        for b_r in range(3):
            a_c = a_col_splits[b_r]
            b_r_start = b_row_splits[b_r]
            b_c_start = b_col_splits[b_c]
            
            if b_r < 2:
                # 2x2 block
                col = add_strassen_block(0, a_c, b_r_start, b_c_start, col)
            else:
                # 2x1 x 1x2 block
                col = add_outer_block(0, a_c, b_r_start, b_c_start, col)
                
    return {"U": U, "V": V, "W": W}
