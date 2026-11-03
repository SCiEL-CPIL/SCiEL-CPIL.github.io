import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    # We are building a rank 84 decomposition.
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2 multiplication
    # X is 2x2, Y is 2x2, Z = XY
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define the 7 bilinear forms as tuples of (x_indices, y_indices, coeffs)
    # x_indices are (i,j) in 2x2, y_indices are (i,j) in 2x2
    # coeff is the coefficient in the product
    
    strassen_forms = [
        # M1
        { 'x': [(0,0,1), (1,1,1)], 'y': [(0,0,1), (1,1,1)], 'z': [(0,0,1), (1,1,-1)] }, # Wait, Z coeffs need to be mapped
    ]
    # This structure is getting complex. Let's just iterate and fill U, V, W directly.
    
    # Helper to add a term
    def add_term(x_indices, y_indices, z_indices, coeff=1):
        nonlocal col
        if col >= rank:
            return
        for (xi, xj, cx) in x_indices:
            U[xi * m + xj][col] += coeff * cx
        for (yi, yj, cy) in y_indices:
            V[yi * p + yj][col] += coeff * cy
        for (zi, zj, cz) in z_indices:
            # W index is k*n + i. Here z is C. C[i,k].
            # So index is zj * n + zi.
            W[zj * n + zi][col] += coeff * cz
        col += 1

    # Iterate over 3 blocks of B columns
    for b_block in range(3):
        b_col_start = b_block * 2
        # B rows are 0..3. Split into top (0,1) and bot (2,3)
        # A cols are 0..3. Split into left (0,1) and right (2,3)
        # A rows are 0..3. Split into top (0,1) and bot (2,3)
        
        # 4 combinations of A block and B row block
        # 1. A_top_left (rows 0,1 cols 0,1) * B_top (rows 0,1 cols b_col_start, b_col_start+1)
        #    Result adds to C_top (rows 0,1 cols b_col_start, b_col_start+1)
        
        # Define A block ranges
        # A_top_left: i in 0,1; j in 0,1
        # A_top_right: i in 0,1; j in 2,3
        # A_bot_left: i in 2,3; j in 0,1
        # A_bot_right: i in 2,3; j in 2,3
        
        # B_top: j in 0,1; k in b_col_start, b_col_start+1
        # B_bot: j in 2,3; k in b_col_start, b_col_start+1
        
        # C_top: i in 0,1; k in b_col_start, b_col_start+1
        # C_bot: i in 2,3; k in b_col_start, b_col_start+1
        
        # We need to map local 2x2 indices to global indices.
        
        def get_strassen_terms(x_i_offset, x_j_offset, y_i_offset, y_j_offset, z_i_offset, z_j_offset):
            # x_i, x_j are local 0,1. Global i = x_i_offset + x_i.
            # y_i, y_j are local 0,1. Global j = y_i_offset + y_i.
            # z_i, z_j are local 0,1. Global i = z_i_offset + z_i. Global k = z_j_offset + z_j.
            
            # M1
            add_term(
                [(x_i_offset+0, x_j_offset+0, 1), (x_i_offset+1, x_j_offset+1, 1)],
                [(y_i_offset+0, y_j_offset+0, 1), (y_i_offset+1, y_j_offset+1, 1)],
                [(z_i_offset+0, z_j_offset+0, 1), (z_i_offset+1, z_j_offset+1, -1)]
            )
            # M2
            add_term(
                [(x_i_offset+1, x_j_offset+0, 1), (x_i_offset+1, x_j_offset+1, 1)],
                [(y_i_offset+0, y_j_offset+0, 1)],
                [(z_i_offset+1, z_j_offset+0, 1)]
            )
            # M3
            add_term(
                [(x_i_offset+0, x_j_offset+0, 1)],
                [(y_i_offset+0, y_j_offset+1, 1), (y_i_offset+1, y_j_offset+1, -1)],
                [(z_i_offset+0, z_j_offset+1, 1)]
            )
            # M4
            add_term(
                [(x_i_offset+1, x_j_offset+1, 1)],
                [(y_i_offset+1, y_j_offset+0, 1), (y_i_offset+0, y_j_offset+0, -1)],
                [(z_i_offset+0, z_j_offset+0, 1), (z_i_offset+1, z_j_offset+0, 1)]
            )
            # M5
            add_term(
                [(x_i_offset+0, x_j_offset+0, 1), (x_i_offset+0, x_j_offset+1, 1)],
                [(y_i_offset+1, y_j_offset+1, 1)],
                [(z_i_offset+0, z_j_offset+0, -1), (z_i_offset+0, z_j_offset+1, 1)]
            )
            # M6
            add_term(
                [(x_i_offset+1, x_j_offset+0, 1), (x_i_offset+0, x_j_offset+0, -1)],
                [(y_i_offset+0, y_j_offset+0, 1), (y_i_offset+0, y_j_offset+1, 1)],
                [(z_i_offset+1, z_j_offset+1, 1)]
            )
            # M7
            add_term(
                [(x_i_offset+0, x_j_offset+1, 1), (x_i_offset+1, x_j_offset+1, -1)],
                [(y_i_offset+1, y_j_offset+0, 1), (y_i_offset+1, y_j_offset+1, 1)],
                [(z_i_offset+0, z_j_offset+0, 1), (z_i_offset+1, z_j_offset+1, 1)]
            )

        # 1. A_top_left * B_top -> C_top
        # x_off: i=0, j=0. y_off: j=0, k=b_col_start. z_off: i=0, k=b_col_start
        get_strassen_terms(0, 0, 0, b_col_start, 0, b_col_start)
        
        # 2. A_top_right * B_bot -> C_top
        # x_off: i=0, j=2. y_off: j=2, k=b_col_start. z_off: i=0, k=b_col_start
        get_strassen_terms(0, 2, 2, b_col_start, 0, b_col_start)
        
        # 3. A_bot_left * B_top -> C_bot
        # x_off: i=2, j=0. y_off: j=0, k=b_col_start. z_off: i=2, k=b_col_start
        get_strassen_terms(2, 0, 0, b_col_start, 2, b_col_start)
        
        # 4. A_bot_right * B_bot -> C_bot
        # x_off: i=2, j=2. y_off: j=2, k=b_col_start. z_off: i=2, k=b_col_start
        get_strassen_terms(2, 2, 2, b_col_start, 2, b_col_start)

    return {"U": U, "V": V, "W": W}
