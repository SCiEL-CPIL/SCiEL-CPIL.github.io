"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx, u_val; v_idx, v_val; w_idx, w_val
    # But Strassen terms have multiple non-zeros in U, V, W
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for ui, uv in zip(u_indices, u_vals):
            U[ui][col] += uv
        for vi, vv in zip(v_indices, v_vals):
            V[vi][col] += vv
        for wi, wv in zip(w_indices, w_vals):
            W[wi][col] += wv
        col += 1

    # Part 1: A1 (2x2) x B1 (2x6)
    # Split B1 into 3 blocks of 2x2
    # Global i: 0,1
    # Global j: 0,1
    # Global k: 0..5
    
    # Strassen 2x2x2 terms
    # P1: (a11+a22)(b11+b22) -> C11, C22
    # P2: (a21+a22)b11 -> C21
    # P3: a11(b12-b22) -> C12
    # P4: a22(b21-b11) -> C11, C21
    # P5: (a11+a12)b22 -> C12, C11
    # P6: (a21-a11)(b11+b12) -> C22
    # P7: (a12-a22)(b21+b22) -> C11, C22
    
    # Local indices for 2x2x2:
    # i: 0,1; j: 0,1; k: 0,1
    # U idx: i*2 + j
    # V idx: j*2 + k
    # W idx: k*2 + i
    
    strassen_terms = [
        # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        # P1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2
        ([2, 3], [1, 1], [0], [1], [1], [1]), # C21 -> k=1, i=1? No.
        # C21 is i=1, k=0? No.
        # C indices: i is row, k is col.
        # C21 means row 2, col 1. i=1 (0-idx), k=0 (0-idx).
        # W idx: k*n + i = 0*2 + 1 = 1.
        # P2 contributes to C21 only.
        # P2 = (a21+a22)b11.
        # a21: i=1, j=0 -> U[2]. a22: i=1, j=1 -> U[3].
        # b11: j=0, k=0 -> V[0].
        # C21: i=1, k=0 -> W[1].
        # So:
        ([2, 3], [1, 1], [0], [1], [1], [1]),
        # P3: a11(b12-b22) -> C12
        # a11: U[0].
        # b12: j=0, k=1 -> V[1]. b22: j=1, k=1 -> V[3].
        # C12: i=0, k=1 -> W[2].
        ([0], [1], [1, 3], [1, -1], [2], [1]),
        # P4: a22(b21-b11) -> C11, C21
        # a22: U[3].
        # b21: j=1, k=0 -> V[2]. b11: j=0, k=0 -> V[0].
        # C11: i=0, k=0 -> W[0].
        # C21: i=1, k=0 -> W[1].
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        # P5: (a11+a12)b22 -> C12, C11
        # a11: U[0]. a12: U[1].
        # b22: j=1, k=1 -> V[3].
        # C12: W[2]. C11: W[0].
        ([0, 1], [1, 1], [3], [1], [2, 0], [1, 1]),
        # P6: (a21-a11)(b11+b12) -> C22
        # a21: U[2]. a11: U[0].
        # b11: V[0]. b12: V[1].
        # C22: i=1, k=1 -> W[3].
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # P7: (a12-a22)(b21+b22) -> C11, C22
        # a12: U[1]. a22: U[3].
        # b21: V[2]. b22: V[3].
        # C11: W[0]. C22: W[3].
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 3], [1, 1]),
    ]
    
    # Apply Part 1
    for block in range(3):
        # k offset
        k_off = block * 2
        # j offset is 0 (since A1 is first 2 cols)
        j_off = 0
        # i offset is 0
        i_off = 0
        
        for term in strassen_terms:
            ui, uv = term[0], term[1]
            vi, vv = term[2], term[3]
            wi, wv = term[4], term[5]
            
            # Map local to global
            g_ui = [ (i_off + i) * m + (j_off + j) for i, j in zip(ui, ui) ] # Wait, ui are indices into 2x2 matrix.
            # ui are 0,1,2,3 corresponding to i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1.
            # So i = ui // 2, j = ui % 2.
            g_ui = [ (i_off + u//2) * m + (j_off + u%2) for u in ui ]
            
            # vi are 0,1,2,3 corresponding to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1.
            # j = vi // 2, k = vi % 2.
            g_vi = [ (j_off + v//2) * p + (k_off + v%2) for v in vi ]
            
            # wi are 0,1,2,3 corresponding to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1.
            # k = wi // 2, i = wi % 2.
            g_wi = [ (k_off + w//2) * n + (i_off + w%2) for w in wi ]
            
            add_term(g_ui, uv, g_vi, vv, g_wi, wv)

    # Part 2: A2 (2x3) x B2 (3x6)
    # j offset is 2 (cols 3,4,5)
    # Split j into 1 + 2.
    # A2a: col 0 (global j=2). 2x1.
    # A2b: cols 1,2 (global j=3,4). 2x2.
    # B2a: row 0 (global j=2). 1x2.
    # B2b: rows 1,2 (global j=3,4). 2x2.
    
    # For each k-block (3 blocks of 2)
    for block in range(3):
        k_off = block * 2
        j_off = 2
        
        # Sub-part 2a: A2a (2x1) x B2a (1x2)
        # Rank 4.
        # Terms:
        # T1: a11 b11 -> C11
        # T2: a11 b12 -> C12
        # T3: a21 b11 -> C21
        # T4: a21 b12 -> C22
        
        # Local indices:
        # A2a: i=0,1; j=0. U idx: i*1 + 0? No, U is global.
        # We use global indices directly.
        
        # T1:
        # U: i=0, j=2 (global). idx 0*5+2=2.
        # V: j=2 (global), k=0 (local 0). idx 2*6+0=12.
        # W: k=0 (local 0), i=0. idx 0*2+0=0.
        add_term([2], [1], [12], [1], [0], [1])
        
        # T2:
        # U: i=0, j=2. idx 2.
        # V: j=2, k=1. idx 13.
        # W: k=1, i=0. idx 1*2+0=2.
        add_term([2], [1], [13], [1], [2], [1])
        
        # T3:
        # U: i=1, j=2. idx 1*5+2=7.
        # V: j=2, k=0. idx 12.
        # W: k=0, i=1. idx 0*2+1=1.
        add_term([7], [1], [12], [1], [1], [1])
        
        # T4:
        # U: i=1, j=2. idx 7.
        # V: j=2, k=1. idx 13.
        # W: k=1, i=1. idx 1*2+1=3.
        add_term([7], [1], [13], [1], [3], [1])
        
        # Sub-part 2b: A2b (2x2) x B2b (2x2)
        # Strassen 2x2x2.
        # Global j for A2b: 3,4.
        # Global j for B2b: 3,4.
        # Local j for Strassen: 0,1.
        
        for term in strassen_terms:
            ui, uv = term[0], term[1]
            vi, vv = term[2], term[3]
            wi, wv = term[4], term[5]
            
            # Map local Strassen indices to global
            # ui: 0->i=0,j=3; 1->i=0,j=4; 2->i=1,j=3; 3->i=1,j=4
            g_ui = [ (u//2) * m + (3 + u%2) for u in ui ]
            
            # vi: 0->j=3,k=0; 1->j=3,k=1; 2->j=4,k=0; 3->j=4,k=1
            g_vi = [ (3 + v//2) * p + (k_off + v%2) for v in vi ]
            
            # wi: 0->k=0,i=0; 1->k=0,i=1; 2->k=1,i=0; 3->k=1,i=1
            g_wi = [ (k_off + w//2) * n + (w%2) for w in wi ]
            
            add_term(g_ui, uv, g_vi, vv, g_wi, wv)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
