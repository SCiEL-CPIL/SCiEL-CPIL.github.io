import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    # Check indices:
    # M1: A00+A11 -> 1,0,0,1 (cols 0)
    # M2: A10+A11 -> 0,0,1,1 (cols 1)
    # M3: A00 -> 1,0,0,0 (cols 2)
    # M4: A11 -> 0,0,0,1 (cols 3)
    # M5: A00+A01 -> 1,1,0,0 (cols 4)
    # M6: A10-A00 -> -1,0,1,0 (cols 5)
    # M7: A01-A11 -> 0,1,0,-1 (cols 6)
    # My U_strassen above is transposed or wrong?
    # Let's re-verify U_strassen rows.
    # Row 0 (A00): 1, 0, 1, 0, 1, -1, 0
    # Row 1 (A01): 0, 0, 0, 0, 1, 0, 1
    # Row 2 (A10): 0, 1, 0, 0, 0, 1, 0
    # Row 3 (A11): 1, 1, 0, 1, 0, 0, -1
    # This looks correct.

    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    # Row 0 (B00): 1, 1, 0, -1, 0, 1, 0
    # Row 1 (B01): 0, 0, 1, 0, 0, 1, 0
    # Row 2 (B10): 0, 0, -1, 1, 0, 0, 1
    # Row 3 (B11): 1, 0, 0, 0, 1, 0, 1

    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    # Row 0 (C00): 1, 0, 0, 1, -1, 0, 1
    # Row 1 (C01): 0, 0, 1, 0, 1, 0, 0
    # Row 2 (C10): 0, 1, 0, 1, 0, 0, 0
    # Row 3 (C11): 1, -1, 1, 0, 0, 1, 0

    # Construct 4x4x4 factors (rank 56)
    # 8 products:
    # P1: A11 B11
    # P2: A12 B21
    # P3: A11 B12
    # P4: A12 B22
    # P5: A21 B11
    # P6: A22 B21
    # P7: A21 B12
    # P8: A22 B22
    
    # A indices: i*4+j
    # B indices: j*4+k
    # C indices: k*4+i
    
    # We will build U_444 (16x56), V_444 (16x56), W_444 (16x56)
    
    U_444 = [[0]*56 for _ in range(16)]
    V_444 = [[0]*56 for _ in range(16)]
    W_444 = [[0]*56 for _ in range(16)]
    
    products = [
        (0, 0, 0, 0), # A11 B11 -> A block 0,0; B block 0,0
        (0, 1, 1, 0), # A12 B21 -> A block 0,1; B block 1,0
        (0, 0, 0, 1), # A11 B12 -> A block 0,0; B block 0,1
        (0, 1, 1, 1), # A12 B22 -> A block 0,1; B block 1,1
        (1, 0, 0, 0), # A21 B11 -> A block 1,0; B block 0,0
        (1, 1, 1, 0), # A22 B21 -> A block 1,1; B block 1,0
        (1, 0, 0, 1), # A21 B12 -> A block 1,0; B block 0,1
        (1, 1, 1, 1), # A22 B22 -> A block 1,1; B block 1,1
    ]
    
    # Mapping of block (bi, bj) to row offsets
    # A block (bi, bj) rows: bi*4 + bj*2 ... bi*4 + bj*2 + 1, bi*4 + bj*2 + 2, bi*4 + bj*2 + 3?
    # A is 4x4. Block (0,0) is rows 0,1 cols 0,1. Indices 0,1,4,5.
    # This is not contiguous.
    # Let's just iterate over the 4x4 indices.
    
    for p_idx, (ai, aj, bi, bj) in enumerate(products):
        col_offset = p_idx * 7
        # A block (ai, aj)
        # Rows in A: ai*2 + 0, ai*2 + 1, (ai*2+1)*4 + ...?
        # A index = i*4 + j.
        # Block (ai, aj) covers i in [ai*2, ai*2+1], j in [aj*2, aj*2+1].
        a_rows = [ (ai*2 + di)*4 + (aj*2 + dj) for di in range(2) for dj in range(2) ]
        # a_rows will be [i0j0, i0j1, i1j0, i1j1]
        
        # B block (bi, bj)
        # B index = j*4 + k.
        # Block (bi, bj) covers j in [bi*2, bi*2+1], k in [bj*2, bj*2+1].
        b_rows = [ (bi*2 + dj)*4 + (bj*2 + dk) for dj in range(2) for dk in range(2) ]
        # b_rows will be [j0k0, j0k1, j1k0, j1k1]
        
        # C block (bj, ai) ?
        # C = A B. C index = k*4 + i.
        # C block corresponding to A block (ai, aj) and B block (bi, bj)?
        # The product A_block(ai,aj) * B_block(bi,bj) contributes to C block (ai, bj).
        # C block (ai, bj) covers i in [ai*2, ai*2+1], k in [bj*2, bj*2+1].
        c_rows = [ (bj*2 + dk)*4 + (ai*2 + di) for dk in range(2) for di in range(2) ]
        
        for r in range(4):
            U_444[a_rows[r]][col_offset + r] = U_strassen[r][0] # Wait, U_strassen is 4x7.
            # U_strassen[r][c] is the coefficient for A_row r in product c.
            # So for product p_idx, column col_offset + c, row a_rows[r] is U_strassen[r][c].
            
        for c in range(7):
            for r in range(4):
                U_444[a_rows[r]][col_offset + c] = U_strassen[r][c]
                V_444[b_rows[r]][col_offset + c] = V_strassen[r][c]
                W_444[c_rows[r]][col_offset + c] = W_strassen[r][c]

    # Now for 4x4x8
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # C1 = A B1, C2 = A B2
    # U_total: 16x112. First 56 cols = U_444. Next 56 cols = U_444.
    # V_total: 32x112.
    #   Rows 0-15 (B1): First 56 cols = V_444. Next 56 cols = 0.
    #   Rows 16-31 (B2): First 56 cols = 0. Next 56 cols = V_444.
    # W_total: 32x112.
    #   Rows 0-15 (C1): First 56 cols = W_444. Next 56 cols = 0.
    #   Rows 16-31 (C2): First 56 cols = 0. Next 56 cols = W_444.
    
    U_total = [[0]*112 for _ in range(16)]
    V_total = [[0]*112 for _ in range(32)]
    W_total = [[0]*112 for _ in range(32)]
    
    for i in range(16):
        for c in range(56):
            U_total[i][c] = U_444[i][c]
            U_total[i][c+56] = U_444[i][c]
            
            V_total[i][c] = V_444[i][c]
            # V_total[i][c+56] = 0 (already 0)
            
            W_total[i][c] = W_444[i][c]
            # W_total[i][c+56] = 0 (already 0)
            
            # For B2 and C2 (rows 16-31)
            V_total[i+16][c+56] = V_444[i][c]
            W_total[i+16][c+56] = W_444[i][c]
            
    return {"U": U_total, "V": V_total, "W": W_total}
