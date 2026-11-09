import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx, v_idx, w_idx are (row, col) in local 2x2 or 2x1/1x4 blocks
        # Need to map to global indices
        pass

    # We will manually fill the arrays based on the Strassen + Schoolbook logic
    # Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Schoolbook: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]

    # Helper to add a rank-1 term
    # u_coords: list of (i, j, val) for A
    # v_coords: list of (j, k, val) for B
    # w_coords: list of (k, i, val) for C
    def add_rank1_term(col, u_entries, v_entries, w_entries):
        for (i, j, val) in u_entries:
            U[i * m + j][col] += val
        for (j, k, val) in v_entries:
            V[j * p + k][col] += val
        for (k, i, val) in w_entries:
            W[k * n + i][col] += val

    col = 0
    
    # Strassen for 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # a_row_offset, a_col_offset: offset in A
        # b_row_offset, b_col_offset: offset in B
        # c_row_offset, c_col_offset: offset in C
        # Returns next col index
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [
            (a_row_offset+0, a_col_offset+0, 1),
            (a_row_offset+1, a_col_offset+1, 1)
        ]
        v1 = [
            (b_row_offset+0, b_col_offset+0, 1),
            (b_row_offset+1, b_col_offset+1, 1)
        ]
        w1 = [
            (c_row_offset+0, c_col_offset+0, 1), # C00
            (c_row_offset+1, c_col_offset+1, 1)  # C11
        ]
        # Note: W index is (k, i) in global? No, W[k*n+i].
        # C[i, k] gets w_val.
        # So w_entries are (k, i, val).
        # C00 is i=0, k=0 -> (0, 0, 1)
        # C11 is i=1, k=1 -> (1, 1, 1)
        # But offsets!
        # i_global = c_row_offset + i_local
        # k_global = c_col_offset + k_local
        # So w_entries should be (k_global, i_global, val)
        
        w1 = [
            (c_col_offset+0, c_row_offset+0, 1),
            (c_col_offset+1, c_row_offset+1, 1)
        ]
        add_rank1_term(start_col, u1, v1, w1)
        col = start_col + 1

        # M2 = (A10+A11)B00
        u2 = [
            (a_row_offset+1, a_col_offset+0, 1),
            (a_row_offset+1, a_col_offset+1, 1)
        ]
        v2 = [
            (b_row_offset+0, b_col_offset+0, 1)
        ]
        w2 = [
            (c_col_offset+0, c_row_offset+1, 1) # C10
        ]
        add_rank1_term(col, u2, v2, w2)
        col += 1

        # M3 = A00(B01-B11)
        u3 = [
            (a_row_offset+0, a_col_offset+0, 1)
        ]
        v3 = [
            (b_row_offset+0, b_col_offset+1, 1),
            (b_row_offset+1, b_col_offset+1, -1)
        ]
        w3 = [
            (c_col_offset+1, c_row_offset+0, 1) # C01
        ]
        add_rank1_term(col, u3, v3, w3)
        col += 1

        # M4 = A11(B10-B00)
        u4 = [
            (a_row_offset+1, a_col_offset+1, 1)
        ]
        v4 = [
            (b_row_offset+1, b_col_offset+0, 1),
            (b_row_offset+0, b_col_offset+0, -1)
        ]
        w4 = [
            (c_col_offset+0, c_row_offset+0, 1), # C00
            (c_col_offset+0, c_row_offset+1, 1)  # C10
        ]
        add_rank1_term(col, u4, v4, w4)
        col += 1

        # M5 = (A00+A01)B11
        u5 = [
            (a_row_offset+0, a_col_offset+0, 1),
            (a_row_offset+0, a_col_offset+1, 1)
        ]
        v5 = [
            (b_row_offset+1, b_col_offset+1, 1)
        ]
        w5 = [
            (c_col_offset+0, c_row_offset+0, -1), # C00
            (c_col_offset+1, c_row_offset+0, 1)   # C01
        ]
        add_rank1_term(col, u5, v5, w5)
        col += 1

        # M6 = (A10-A00)(B00+B01)
        u6 = [
            (a_row_offset+1, a_col_offset+0, 1),
            (a_row_offset+0, a_col_offset+0, -1)
        ]
        v6 = [
            (b_row_offset+0, b_col_offset+0, 1),
            (b_row_offset+0, b_col_offset+1, 1)
        ]
        w6 = [
            (c_col_offset+1, c_row_offset+1, 1) # C11
        ]
        add_rank1_term(col, u6, v6, w6)
        col += 1

        # M7 = (A01-A11)(B10+B11)
        u7 = [
            (a_row_offset+0, a_col_offset+1, 1),
            (a_row_offset+1, a_col_offset+1, -1)
        ]
        v7 = [
            (b_row_offset+1, b_col_offset+0, 1),
            (b_row_offset+1, b_col_offset+1, 1)
        ]
        w7 = [
            (c_col_offset+0, c_row_offset+0, 1)  # C00
        ]
        add_rank1_term(col, u7, v7, w7)
        col += 1

        return col

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = strassen_2x2(0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # B rows 0,1. Cols 2,3.
    # B local (0,0) is Global (0,2).
    # B local (1,0) is Global (1,2).
    # B local (0,1) is Global (0,3).
    # B local (1,1) is Global (1,3).
    # So b_row_offset=0, b_col_offset=2.
    col = strassen_2x2(0, 0, 0, 2, 0, 2, col)

    # Schoolbook for A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # A col 2. B row 2.
    for i in range(2):
        for k in range(4):
            u_sb = [(i, 2, 1)]
            v_sb = [(2, k, 1)]
            w_sb = [(k, i, 1)]
            add_rank1_term(col, u_sb, v_sb, w_sb)
            col += 1
            
    return {"U": U, "V": V, "W": W}
