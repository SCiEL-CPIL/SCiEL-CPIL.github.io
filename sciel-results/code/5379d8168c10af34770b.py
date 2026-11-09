import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        if col >= rank:
            return
        for u_idx, u_val in zip(u_indices, u_vals):
            U[u_idx][col] += u_val
        for v_idx, v_val in zip(v_indices, v_vals):
            V[v_idx][col] += v_val
        for w_idx, w_val in zip(w_indices, w_vals):
            W[w_idx][col] += w_val
        col += 1

    # Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        # A is 2x2: rows i_off, i_off+1; cols j_off, j_off+1
        # B is 2x2: rows j_off, j_off+1; cols k_off, k_off+1
        # C is 2x2: rows i_off, i_off+1; cols k_off, k_off+1
        
        # Indices for U (A matrix, n*m = 12 entries): row*4 + col
        a00 = i_off * 4 + j_off
        a01 = i_off * 4 + j_off + 1
        a10 = (i_off + 1) * 4 + j_off
        a11 = (i_off + 1) * 4 + j_off + 1
        
        # Indices for V (B matrix, m*p = 16 entries): row*4 + col
        b00 = j_off * 4 + k_off
        b01 = j_off * 4 + k_off + 1
        b10 = (j_off + 1) * 4 + k_off
        b11 = (j_off + 1) * 4 + k_off + 1
        
        # Indices for W (C matrix, p*n = 12 entries): col*3 + row
        c00 = k_off * 3 + i_off
        c01 = (k_off + 1) * 3 + i_off
        c10 = k_off * 3 + (i_off + 1)
        c11 = (k_off + 1) * 3 + (i_off + 1)
        
        # M1 = (A00+A11)(B00+B11)
        add_rank1_term([a00, a11], [b00, b11], [c00, c11], [1, 1], [1, 1], [1, 1])
        # M2 = (A10+A11)B00
        add_rank1_term([a10, a11], [b00], [c10, c11], [1, 1], [1], [1, -1])
        # M3 = A00(B01-B11)
        add_rank1_term([a00], [b01, b11], [c01, c11], [1], [1, -1], [1, 1])
        # M4 = A11(B10-B00)
        add_rank1_term([a11], [b10, b00], [c00, c10], [1], [1, -1], [1, 1])
        # M5 = (A00+A01)B11
        add_rank1_term([a00, a01], [b11], [c00, c01], [1, 1], [1], [-1, 1])
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term([a10, a00], [b00, b01], [c11], [1, -1], [1, 1], [1])
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term([a01, a11], [b10, b11], [c00], [1, -1], [1, 1], [1])

    # Schoolbook 1x2x2
    def add_schoolbook_1x2x2(i_idx, j_off, k_off):
        # A is 1x2: row i_idx; cols j_off, j_off+1
        # B is 2x2: rows j_off, j_off+1; cols k_off, k_off+1
        # C is 1x2: row i_idx; cols k_off, k_off+1
        
        a0 = i_idx * 4 + j_off
        a1 = i_idx * 4 + j_off + 1
        
        b00 = j_off * 4 + k_off
        b01 = j_off * 4 + k_off + 1
        b10 = (j_off + 1) * 4 + k_off
        b11 = (j_off + 1) * 4 + k_off + 1
        
        c0 = k_off * 3 + i_idx
        c1 = (k_off + 1) * 3 + i_idx
        
        # T1: A0 B00 C0
        add_rank1_term([a0], [b00], [c0], [1], [1], [1])
        # T2: A1 B10 C0
        add_rank1_term([a1], [b10], [c0], [1], [1], [1])
        # T3: A0 B01 C1
        add_rank1_term([a0], [b01], [c1], [1], [1], [1])
        # T4: A1 B11 C1
        add_rank1_term([a1], [b11], [c1], [1], [1], [1])

    # Block 1: A_top (2x4) * B_left (4x2) -> C_top_left (2x2)
    # A_top rows 0,1; B_left cols 0,1; C_top_left rows 0,1, cols 0,1
    add_strassen(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2]
    
    # Block 2: A_top (2x4) * B_right (4x2) -> C_top_right (2x2)
    # A_top rows 0,1; B_right cols 2,3; C_top_right rows 0,1, cols 2,3
    add_strassen(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4]
    
    # Block 3: A_bot (1x4) * B_left (4x2) -> C_bot_left (1x2)
    # A_bot row 2; B_left cols 0,1; C_bot_left row 2, cols 0,1
    add_schoolbook_1x2x2(2, 0, 0) # A[2, 0:2] * B[0:2, 0:2]
    add_schoolbook_1x2x2(2, 2, 0) # A[2, 2:4] * B[2:4, 0:2]
    
    # Block 4: A_bot (1x4) * B_right (4x2) -> C_bot_right (1x2)
    # A_bot row 2; B_right cols 2,3; C_bot_right row 2, cols 2,3
    add_schoolbook_1x2x2(2, 0, 2) # A[2, 0:2] * B[0:2, 2:4]
    add_schoolbook_1x2x2(2, 2, 2) # A[2, 2:4] * B[2:4, 2:4]
    
    return {"U": U, "V": V, "W": W}
