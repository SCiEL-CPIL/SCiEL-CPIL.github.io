def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Base Strassen vectors for 2x2x2
    # u: [a, b, c, d]
    # v: [e, f, g, h]
    # w: [C11, C12, C21, C22]
    
    u_base = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v_base = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_base = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Helper to add a block
    # sub: 0 for Left, 1 for Right
    # block: 0, 1, 2 (corresponding to B col pairs)
    # col_offset: starting column in U,V,W
    def add_block(sub, block, col_offset):
        # U mapping
        # A_L: rows 0-3, A_R: rows 4-7
        u_row_offset = sub * 4
        
        # V mapping
        # B_L: rows 0-11, B_R: rows 12-23
        # Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5.
        # e,f from row 0 of B_block, g,h from row 1 of B_block.
        # Global V index: j*p + k.
        # For B_L (sub=0): j=0,1. For B_R (sub=1): j=2,3.
        # b_row_0 = sub * 2 * p + block * 2
        # b_row_1 = (sub * 2 + 1) * p + block * 2
        b_row_0 = sub * 2 * p + block * 2
        b_row_1 = (sub * 2 + 1) * p + block * 2
        
        # W mapping
        # C cols: block 0: 0,1. block 1: 2,3. block 2: 4,5.
        # W index: k*n + i.
        # C11: k=block*2, i=0 -> (block*2)*2 + 0
        # C12: k=block*2+1, i=0 -> (block*2+1)*2 + 0
        # C21: k=block*2, i=1 -> (block*2)*2 + 1
        # C22: k=block*2+1, i=1 -> (block*2+1)*2 + 1
        w_idx_11 = (block * 2) * n + 0
        w_idx_12 = (block * 2 + 1) * n + 0
        w_idx_21 = (block * 2) * n + 1
        w_idx_22 = (block * 2 + 1) * n + 1

        for r in range(7):
            col = col_offset + r
            
            # U
            for i in range(4):
                U[u_row_offset + i][col] = u_base[r][i]
                
            # V
            # v_base[r] = [e, f, g, h]
            # e -> b_row_0
            # f -> b_row_0 + 1
            # g -> b_row_1
            # h -> b_row_1 + 1
            V[b_row_0][col] = v_base[r][0]
            V[b_row_0 + 1][col] = v_base[r][1]
            V[b_row_1][col] = v_base[r][2]
            V[b_row_1 + 1][col] = v_base[r][3]
            
            # W
            W[w_idx_11][col] = w_base[r][0]
            W[w_idx_12][col] = w_base[r][1]
            W[w_idx_21][col] = w_base[r][2]
            W[w_idx_22][col] = w_base[r][3]

    # Sub-problem 1 (Left)
    for block in range(3):
        add_block(0, block, block * 7)
        
    # Sub-problem 2 (Right)
    for block in range(3):
        add_block(1, block, 21 + block * 7)

    return {"U": U, "V": V, "W": W}
