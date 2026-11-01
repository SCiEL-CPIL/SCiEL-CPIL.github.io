"""Program for 2x3x7 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(u_vec, v_vec, w_vec):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_vec:
            U_col[idx] = coeff
        for idx, coeff in v_vec:
            V_col[idx] = coeff
        for idx, coeff in w_vec:
            W_col[idx] = coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Part 1: A2 (2x1) * B2 (1x7)
    # A2 is col 2 of A. B2 is row 2 of B.
    for i in range(n):
        for k in range(p):
            u = [(u_idx(i, 2), 1.0)]
            v = [(v_idx(2, k), 1.0)]
            w = [(w_idx(k, i), 1.0)]
            add_decomp(u, v, w)

    # Part 2: A1 (2x2) * B1 (2x7)
    # A1 is cols 0,1 of A. B1 is rows 0,1 of B.
    
    def strassen_decomp(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # M1
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        # M2
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1), (w_idx(b_col_offset+1, a_row_offset+1), -1)]
        add_decomp(u, v, w)
        
        # M3
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+1), 1), (v_idx(b_row_offset+1, b_col_offset+1), -1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        # M4
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+0), -1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        # M5
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), -1), (w_idx(b_col_offset+1, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        
        # M6
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+0), -1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        # M7
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1), (u_idx(a_row_offset+1, a_col_offset+1), -1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

    def schoolbook_221(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # C0 = A00*B00 + A01*B10
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        
        # C1 = A10*B00 + A11*B10
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)

    # B1 cols 0-1
    strassen_decomp(0, 0, 0, 0)
    # B1 cols 2-3
    strassen_decomp(0, 0, 0, 2)
    # B1 cols 4-5
    strassen_decomp(0, 0, 0, 4)
    # B1 col 6
    schoolbook_221(0, 0, 0, 6)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
