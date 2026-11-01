import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # m=5 is split into 2, 2, 1.
    # p=6 is split into 2, 2, 2.
    # n=2 is fixed.
    
    # Total rank
    # 6 blocks of 2x2x2 (Strassen, rank 7)
    # 3 blocks of 2x1x2 (Rank 4)
    rank = 6 * 7 + 3 * 4
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define block boundaries
    # m splits: 0-1, 2-3, 4
    m_splits = [(0, 2), (2, 4), (4, 5)]
    # p splits: 0-1, 2-3, 4-5
    p_splits = [(0, 2), (2, 4), (4, 6)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_start, a_end, b_start_row, b_end_row, b_start_col, b_end_col):
        nonlocal col
        # a_start, a_end: indices in A (cols)
        # b_start_row, b_end_row: indices in B (rows)
        # b_start_col, b_end_col: indices in B (cols)
        
        # Local indices
        # A is 2x2
        # B is 2x2
        
        # M1
        U[a_start * 1 + 0][col] = 1  # A00
        U[a_end * 1 + 1][col] = 1    # A11 (a_end is exclusive, so last index is a_end-1)
        # Wait, A is 2x5. Row i, col j.
        # U index: i*m + j.
        # For 2x2 block of A:
        # i=0, j=a_start -> U[0*m + a_start]
        # i=1, j=a_start -> U[1*m + a_start]
        # i=0, j=a_start+1 -> U[0*m + a_start+1]
        # i=1, j=a_start+1 -> U[1*m + a_start+1]
        
        # Let's define helper to get U index
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # M1: (A00+A11)(B00+B11)
        U[u_idx(0, a_start)][col] = 1
        U[u_idx(1, a_start+1)][col] = 1
        V[v_idx(b_start_row, b_start_col)][col] = 1
        V[v_idx(b_start_row+1, b_start_col+1)][col] = 1
        W[w_idx(b_start_col, 0)][col] = 1
        W[w_idx(b_start_col+1, 1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_idx(1, a_start)][col] = 1
        U[u_idx(1, a_start+1)][col] = 1
        V[v_idx(b_start_row, b_start_col)][col] = 1
        W[w_idx(b_start_col, 1)][col] = 1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_idx(0, a_start)][col] = 1
        V[v_idx(b_start_row, b_start_col+1)][col] = 1
        V[v_idx(b_start_row+1, b_start_col+1)][col] = -1
        W[w_idx(b_start_col+1, 0)][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_idx(1, a_start+1)][col] = 1
        V[v_idx(b_start_row+1, b_start_col)][col] = 1
        V[v_idx(b_start_row, b_start_col)][col] = -1
        W[w_idx(b_start_col, 1)][col] = 1
        col += 1
        
        # M5: (A00+A10)B11
        U[u_idx(0, a_start)][col] = 1
        U[u_idx(1, a_start)][col] = 1
        V[v_idx(b_start_row+1, b_start_col+1)][col] = 1
        W[w_idx(b_start_col+1, 0)][col] = 1
        W[w_idx(b_start_col+1, 1)][col] = 1
        col += 1
        
        # M6: (A01-A00)(B00+B10)
        U[u_idx(0, a_start+1)][col] = 1
        U[u_idx(0, a_start)][col] = -1
        V[v_idx(b_start_row, b_start_col)][col] = 1
        V[v_idx(b_start_row+1, b_start_col)][col] = 1
        W[w_idx(b_start_col, 0)][col] = 1
        W[w_idx(b_start_col, 1)][col] = 1
        col += 1
        
        # M7: (A10-A11)(B01+B11)
        U[u_idx(1, a_start)][col] = 1
        U[u_idx(1, a_start+1)][col] = -1
        V[v_idx(b_start_row, b_start_col+1)][col] = 1
        V[v_idx(b_start_row+1, b_start_col+1)][col] = 1
        W[w_idx(b_start_col+1, 1)][col] = 1
        col += 1
        
        # Now add to W
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 is at col-7
        # M2 is at col-6
        # M3 is at col-5
        # M4 is at col-4
        # M5 is at col-3
        # M6 is at col-2
        # M7 is at col-1
        
        m1, m2, m3, m4, m5, m6, m7 = col-7, col-6, col-5, col-4, col-3, col-2, col-1
        
        # C00 (k=b_start_col, i=0)
        W[w_idx(b_start_col, 0)][m1] = 1
        W[w_idx(b_start_col, 0)][m4] = 1
        W[w_idx(b_start_col, 0)][m5] = -1
        W[w_idx(b_start_col, 0)][m7] = 1
        
        # C01 (k=b_start_col+1, i=0)
        W[w_idx(b_start_col+1, 0)][m3] = 1
        W[w_idx(b_start_col+1, 0)][m5] = 1
        
        # C10 (k=b_start_col, i=1)
        W[w_idx(b_start_col, 1)][m2] = 1
        W[w_idx(b_start_col, 1)][m4] = 1
        
        # C11 (k=b_start_col+1, i=1)
        W[w_idx(b_start_col+1, 1)][m1] = 1
        W[w_idx(b_start_col+1, 1)][m2] = -1
        W[w_idx(b_start_col+1, 1)][m3] = 1
        W[w_idx(b_start_col+1, 1)][m6] = 1

    def add_rank4_block(a_start, b_start_row, b_start_col):
        nonlocal col
        # A is 2x1, B is 1x2
        # A indices: (0, a_start), (1, a_start)
        # B indices: (b_start_row, b_start_col), (b_start_row, b_start_col+1)
        
        # M1: A00 B00 -> C00
        U[u_idx(0, a_start)][col] = 1
        V[v_idx(b_start_row, b_start_col)][col] = 1
        W[w_idx(b_start_col, 0)][col] = 1
        col += 1
        
        # M2: A00 B01 -> C01
        U[u_idx(0, a_start)][col] = 1
        V[v_idx(b_start_row, b_start_col+1)][col] = 1
        W[w_idx(b_start_col+1, 0)][col] = 1
        col += 1
        
        # M3: A10 B00 -> C10
        U[u_idx(1, a_start)][col] = 1
        V[v_idx(b_start_row, b_start_col)][col] = 1
        W[w_idx(b_start_col, 1)][col] = 1
        col += 1
        
        # M4: A10 B01 -> C11
        U[u_idx(1, a_start)][col] = 1
        V[v_idx(b_start_row, b_start_col+1)][col] = 1
        W[w_idx(b_start_col+1, 1)][col] = 1
        col += 1

    # Iterate blocks
    for r_idx, (a_start, a_end) in enumerate(m_splits):
        for c_idx, (b_start_col, b_end_col) in enumerate(p_splits):
            b_start_row = a_start
            b_end_row = a_end
            
            if a_end - a_start == 2:
                add_strassen_block(a_start, a_end, b_start_row, b_end_row, b_start_col, b_end_col)
            else:
                add_rank4_block(a_start, b_start_row, b_start_col)
                
    return {"U": U, "V": V, "W": W}
