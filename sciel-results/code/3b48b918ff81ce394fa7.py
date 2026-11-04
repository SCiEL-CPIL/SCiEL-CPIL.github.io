import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (28, rank)
    # W: (p*n, rank) -> (28, rank)
    
    # We will build lists of U, V, W columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2x2 decomposition vectors
    # Local indices: a,b,c in 0..3
    # a: A[0,0], A[0,1], A[1,0], A[1,1]
    # b: B[0,0], B[0,1], B[1,0], B[1,1]
    # c: C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Schoolbook 2x2x1 decomposition vectors
    # A: 2x2, B: 2x1, C: 2x1
    # a: A[0,0], A[0,1], A[1,0], A[1,1]
    # b: B[0,0], B[1,0]
    # c: C[0,0], C[1,0]
    
    schoolbook_2x2x1_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    # Function to map local indices to global indices for U (A matrix)
    # A is 4x4. Global index a = i*4 + j.
    # Block (bi, bj) is 2x2.
    # Local index la in 0..3 corresponds to (li, lj) in 0..1.
    # Global i = bi*2 + li, j = bj*2 + lj.
    def map_a(local_vec, block_row, block_col):
        u = [0] * 16
        for la in range(4):
            li = la // 2
            lj = la % 2
            gi = block_row * 2 + li
            gj = block_col * 2 + lj
            ga = gi * 4 + gj
            u[ga] += local_vec[la]
        return u
    
    # Function to map local indices to global indices for V (B matrix)
    # B is 4x7. Global index b = j*7 + k.
    # For 2x2x2 part: B block is 2x2.
    # Block (bj, bk) where bj is row block (0..1), bk is col block (0..1) within the 4 cols.
    # Local index lb in 0..3 corresponds to (lj, lk) in 0..1.
    # Global j = bj*2 + lj, k = bk*2 + lk.
    def map_b_2x2(local_vec, block_row, block_col):
        v = [0] * 28
        for lb in range(4):
            lj = lb // 2
            lk = lb % 2
            gj = block_row * 2 + lj
            gk = block_col * 2 + lk
            gb = gj * 7 + gk
            v[gb] += local_vec[lb]
        return v

    # Function to map local indices to global indices for W (C matrix)
    # C is 4x7. Global index c = k*4 + i.
    # For 2x2x2 part: C block is 2x2.
    # Block (ck, ci) where ck is col block (0..1), ci is row block (0..1) within the 4 rows.
    # Local index lc in 0..3 corresponds to (lk, li) in 0..1.
    # Global k = ck*2 + lk, i = ci*2 + li.
    def map_w_2x2(local_vec, block_col, block_row):
        w = [0] * 28
        for lc in range(4):
            lk = lc // 2
            li = lc % 2
            gk = block_col * 2 + lk
            gi = block_row * 2 + li
            gc = gk * 4 + gi
            w[gc] += local_vec[lc]
        return w

    # --- Part 1: 4x4x4 (cols 0-3) ---
    # Strassen on 4x4 blocks.
    # The 7 terms correspond to:
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # We need to expand these into global U, V, W.
    # Each term is a 2x2x2 multiplication of linear combinations.
    # We iterate through the 7 Strassen terms for the 4x4 case.
    
    # Define the 7 Strassen blocks for 4x4
    # Each block is defined by:
    # - A components: list of (block_row, block_col, coeff)
    # - B components: list of (block_row, block_col, coeff)
    # - C targets: list of (block_col, block_row, coeff)
    
    strassen_4x4_blocks = [
        # M1
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # M2
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(0, 1, 1), (1, 1, -1)] # C10 (col 0, row 1), C11 (col 1, row 1)
        },
        # M3
        {
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'C': [(1, 0, 1), (1, 1, 1)] # C01 (col 1, row 0), C11 (col 1, row 1)
        },
        # M4
        {
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (0, 1, 1)] # C00 (col 0, row 0), C10 (col 0, row 1)
        },
        # M5
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (1, 0, 1)] # C00 (col 0, row 0), C01 (col 1, row 0)
        },
        # M6
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)] # C11 (col 1, row 1)
        },
        # M7
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)] # C00 (col 0, row 0)
        }
    ]
    
    for block in strassen_4x4_blocks:
        # Construct local U, V, W vectors for the 2x2x2 sub-decomposition
        # The "local" A is the sum of the specified A blocks.
        # The "local" B is the sum of the specified B blocks.
        # The "local" C is the sum of the specified C blocks.
        
        # We add the 7 Strassen terms for this sub-decomposition
        for u_loc, v_loc, w_loc in strassen_terms:
            # Expand u_loc to global U
            u_global = [0] * 16
            for (br, bc, coeff) in block['A']:
                # Map local index to global index for this block
                for la in range(4):
                    li = la // 2
                    lj = la % 2
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    ga = gi * 4 + gj
                    u_global[ga] += u_loc[la] * coeff
            
            # Expand v_loc to global V
            v_global = [0] * 28
            for (br, bc, coeff) in block['B']:
                for lb in range(4):
                    lj = lb // 2
                    lk = lb % 2
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    gb = gj * 7 + gk
                    v_global[gb] += v_loc[lb] * coeff
            
            # Expand w_loc to global W
            w_global = [0] * 28
            for (bc, br, coeff) in block['C']:
                for lc in range(4):
                    lk = lc // 2
                    li = lc % 2
                    gk = bc * 2 + lk
                    gi = br * 2 + li
                    gc = gk * 4 + gi
                    w_global[gc] += w_loc[lc] * coeff
            
            add_term(u_global, v_global, w_global)
            
    # --- Part 2: 4x4x3 (cols 4-6) ---
    # 4 blocks of 2x2x3
    # Block (bi, bj) of A, and B_R block (bi, col_range)
    # B_R is 4x3. Cols 4,5,6.
    # Split B_R into top (rows 0-1) and bottom (rows 2-3).
    # Each is 2x3.
    
    # For each A block (bi, bj) in 0..1:
    for bi in range(2):
        for bj in range(2):
            # B_R block corresponding to row block bi
            # If bi=0, B_R rows 0-1 (global rows 0-1, cols 4-6)
            # If bi=1, B_R rows 2-3 (global rows 2-3, cols 4-6)
            
            # Decompose 2x2x3 into 2x2x2 (cols 4-5) and 2x2x1 (col 6)
            
            # 1. 2x2x2 part (cols 4-5)
            # Use Strassen
            for u_loc, v_loc, w_loc in strassen_terms:
                u_global = [0] * 16
                for la in range(4):
                    li = la // 2
                    lj = la % 2
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    ga = gi * 4 + gj
                    u_global[ga] += u_loc[la]
                
                v_global = [0] * 28
                for lb in range(4):
                    lj = lb // 2
                    lk = lb % 2
                    gj = bi * 2 + lj
                    gk = 4 + lk # Cols 4,5
                    gb = gj * 7 + gk
                    v_global[gb] += v_loc[lb]
                
                w_global = [0] * 28
                for lc in range(4):
                    lk = lc // 2
                    li = lc % 2
                    gk = 4 + lk # Cols 4,5
                    gi = bi * 2 + li
                    gc = gk * 4 + gi
                    w_global[gc] += w_loc[lc]
                
                add_term(u_global, v_global, w_global)
            
            # 2. 2x2x1 part (col 6)
            # Use Schoolbook
            for u_loc, v_loc, w_loc in schoolbook_2x2x1_terms:
                u_global = [0] * 16
                for la in range(4):
                    li = la // 2
                    lj = la % 2
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    ga = gi * 4 + gj
                    u_global[ga] += u_loc[la]
                
                v_global = [0] * 28
                for lb in range(2):
                    lj = lb
                    gk = 6 # Col 6
                    gj = bi * 2 + lj
                    gb = gj * 7 + gk
                    v_global[gb] += v_loc[lb]
                
                w_global = [0] * 28
                for lc in range(2):
                    li = lc
                    gk = 6 # Col 6
                    gi = bi * 2 + li
                    gc = gk * 4 + gi
                    w_global[gc] += w_loc[lc]
                
                add_term(u_global, v_global, w_global)

    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
