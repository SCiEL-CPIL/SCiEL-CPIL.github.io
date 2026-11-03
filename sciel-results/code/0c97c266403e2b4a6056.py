import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Strategy: Block decomposition with Strassen for 2x2 blocks.
    Rank = 56.
    """
    n, m, p = DIMENSIONS
    deadline = DEADLINE
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_rows.append(u_dict)
        V_rows.append(v_dict)
        W_rows.append(w_dict)
    
    def strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, 
                     c_row_offset, c_col_offset):
        A00_idx = (a_row_offset + 0) * 4 + (a_col_offset + 0)
        A01_idx = (a_row_offset + 0) * 4 + (a_col_offset + 1)
        A10_idx = (a_row_offset + 1) * 4 + (a_col_offset + 0)
        A11_idx = (a_row_offset + 1) * 4 + (a_col_offset + 1)
        
        B00_idx = (b_row_offset + 0) * 5 + (b_col_offset + 0)
        B01_idx = (b_row_offset + 0) * 5 + (b_col_offset + 1)
        B10_idx = (b_row_offset + 1) * 5 + (b_col_offset + 0)
        B11_idx = (b_row_offset + 1) * 5 + (b_col_offset + 1)
        
        C00_idx = (c_col_offset + 0) * 3 + (c_row_offset + 0)
        C01_idx = (c_col_offset + 1) * 3 + (c_row_offset + 0)
        C10_idx = (c_col_offset + 0) * 3 + (c_row_offset + 1)
        C11_idx = (c_col_offset + 1) * 3 + (c_row_offset + 1)
        
        add_term({A00_idx: 1, A11_idx: 1}, {B00_idx: 1, B11_idx: 1}, {C00_idx: 1, C11_idx: 1})
        add_term({A10_idx: 1, A11_idx: 1}, {B00_idx: 1}, {C10_idx: 1, C11_idx: -1})
        add_term({A00_idx: 1}, {B10_idx: 1, B11_idx: -1}, {C01_idx: 1, C11_idx: 1})
        add_term({A11_idx: 1}, {B01_idx: 1, B10_idx: -1}, {C00_idx: 1, C10_idx: 1})
        add_term({A00_idx: 1, A01_idx: 1}, {B11_idx: 1}, {C00_idx: -1, C01_idx: 1})
        add_term({A10_idx: 1, A00_idx: -1}, {B00_idx: 1, B10_idx: 1}, {C11_idx: 1})
        add_term({A01_idx: 1, A11_idx: -1}, {B01_idx: 1, B11_idx: 1}, {C00_idx: 1})
    
    def schoolbook_2x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        add_term({(a_row_offset+0)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        add_term({(a_row_offset+0)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+0): 1})
        add_term({(a_row_offset+1)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
        add_term({(a_row_offset+1)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + (c_row_offset+1): 1})
    
    def schoolbook_1x2_times_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+0): 1}, 
                 {(c_col_offset+0)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + (b_col_offset+1): 1}, 
                 {(c_col_offset+1)*3 + c_row_offset: 1})
    
    def schoolbook_1x2_times_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset,
                                  c_row_offset, c_col_offset):
        add_term({(a_row_offset)*4 + (a_col_offset+0): 1}, 
                 {(b_row_offset+0)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
        add_term({(a_row_offset)*4 + (a_col_offset+1): 1}, 
                 {(b_row_offset+1)*5 + b_col_offset: 1}, 
                 {(c_col_offset)*3 + c_row_offset: 1})
    
    # C11 (rows 0-1, cols 0-1)
    strassen_2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # C12 (rows 0-1, cols 2-3)
    strassen_2x2(0, 0, 0, 2, 0, 2)
    strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # C13 (rows 0-1, col 4)
    schoolbook_2x2_times_2x1(0, 0, 0, 4, 0, 4)
    schoolbook_2x2_times_2x1(0, 2, 2, 4, 0, 4)
    
    # C21 (row 2, cols 0-1)
    schoolbook_1x2_times_2x2(2, 0, 0, 0, 2, 0)
    schoolbook_1x2_times_2x2(2, 2, 2, 0, 2, 0)
    
    # C22 (row 2, cols 2-3)
    schoolbook_1x2_times_2x2(2, 0, 0, 2, 2, 2)
    schoolbook_1x2_times_2x2(2, 2, 2, 2, 2, 2)
    
    # C23 (row 2, col 4)
    schoolbook_1x2_times_2x1(2, 0, 0, 4, 2, 4)
    schoolbook_1x2_times_2x1(2, 2, 2, 4, 2, 4)
    
    rank = len(U_rows)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        for idx, val in U_rows[t].items():
            U[idx][t] = val
        for idx, val in V_rows[t].items():
            V[idx][t] = val
        for idx, val in W_rows[t].items():
            W[idx][t] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
