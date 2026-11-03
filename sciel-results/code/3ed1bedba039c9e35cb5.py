import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Strassen 2x2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_terms = [
        # A_coeffs, B_coeffs, C_coeffs
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    terms = []
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, scale=1):
        """
        Add 7 Strassen terms for a 2x2x2 block multiplication.
        a_r, a_c: top-left of A block
        b_r, b_c: top-left of B block
        c_r, c_c: top-left of C block
        """
        for A_c, B_c, C_c in strassen_terms:
            u = [0] * 16
            v = [0] * 24
            w = [0] * 24
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    u[(a_r+i)*4 + (a_c+j)] = scale * A_c[i][j]
            
            # Fill V
            for i in range(2):
                for j in range(2):
                    v[(b_r+i)*6 + (b_c+j)] = scale * B_c[i][j]
                    
            # Fill W
            # W index: global_col * n + global_row
            # C block is 2x2 at c_r, c_c
            for i in range(2):
                for j in range(2):
                    w[(c_c+j)*4 + (c_r+i)] = scale * C_c[i][j]
            
            terms.append((u, v, w))

    # 4x4x4 part (cols 0-3 of B, cols 0-3 of C)
    # Split into 2x2 blocks of 2x2 matrices.
    # A blocks: (0,0), (0,2), (2,0), (2,2)
    # B blocks: (0,0), (0,2), (2,0), (2,2)
    # C blocks: (0,0), (0,2), (2,0), (2,2)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    add_strassen_block(0, 0, 0, 0, 0, 0, 1) # A00, B00, C00
    add_strassen_block(0, 0, 2, 2, 0, 0, 1) # A00, B11, C00 (Wait, this is not how Strassen 4x4 works)
    
    # The previous code used `add_product_of_block_sums` which handled sums of blocks.
    # My `add_strassen_block` only handles single blocks.
    # I need to replicate the logic of `add_product_of_block_sums` but cleaner.
    
    pass
