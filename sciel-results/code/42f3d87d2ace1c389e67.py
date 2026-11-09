"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Block decomposition:
    # A (4x5) split into A_top (2x5), A_bot (2x5)
    # B (5x6) split into B_left (5x3), B_right (5x3)
    # C (4x6) has 4 blocks: C_tl(2x3), C_tr(2x3), C_bl(2x3), C_br(2x3)
    #
    # Each C-block = A_sub(2x5) * B_sub(5x3)
    # Split middle dim 5 = 2+3:
    #   A_sub = [A_sub_L(2x2), A_sub_R(2x3)]
    #   B_sub = [B_sub_T(2x3); B_sub_B(3x3)]
    # C-block = A_sub_L * B_sub_T + A_sub_R * B_sub_B
    #
    # For A_sub_L(2x2) * B_sub_T(2x3):
    #   Split B_sub_T into BT_L(2x2) + BT_R(2x1)
    #   Use Strassen on (2x2)(2x2): rank 7
    #   Use schoolbook on (2x2)(2x1): rank 4
    #   Total: 11
    #
    # For A_sub_R(2x3) * B_sub_B(3x3): schoolbook, rank 18
    #
    # Total per C-block: 29
    # Total: 4 * 29 = 116
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. Keys are (row) indices in global space."""
        u_global = [0.0] * (n * m)
        v_global = [0.0] * (m * p)
        w_global = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_global[idx] += val
        for idx, val in v_dict.items():
            v_global[idx] += val
        for idx, val in w_dict.items():
            w_global[idx] += val
        U_list.append(u_global)
        V_list.append(v_global)
        W_list.append(w_global)
    
    def strassen_2x2_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen decomposition for a 2x2 * 2x2 sub-problem.
        A rows: a_row_offset, a_row_offset+1
        B cols: b_col_offset, b_col_offset+1
        C rows: c_row_offset, c_row_offset+1
        C cols: c_col_offset, c_col_offset+1
        
        The 5 middle indices are the same for A cols and B rows.
        """
        # A is 2x2: rows (a_row_offset, a_row_offset+1), cols (0,1) local -> global (0,1)
        # B is 2x2: rows (0,1) local -> global (0,1), cols (b_col_offset, b_col_offset+1)
        # C is 2x2: rows (c_row_offset, c_row_offset+1), cols (c_col_offset, c_col_offset+1)
        
        # W index: k*n + i where k is col index, i is row index
        # For C[i,k], W index = k*n + i
        
        # Strassen products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A11+A21)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A12+A22)B22
        # M6 = (A11-A21)(B21+B22)
        # M7 = (A12-A22)(B11+B21)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: A[i,j] -> U index i*m + j
        # B indices: B[j,k] -> V index j*p + k
        # C indices: C[i,k] -> W index k*n + i
        
        a00 = a_row_offset * m + 0
        a01 = a_row_offset * m + 1
        a10 = (a_row_offset + 1) * m + 0
        a11 = (a_row_offset + 1) * m + 1
        
        b00 = 0 * p + b_col_offset
        b01 = 0 * p + (b_col_offset + 1)
        b10 = 1 * p + b_col_offset
        b11 = 1 * p + (b_col_offset + 1)
        
        c00 = c_col_offset * n + c_row_offset
        c01 = (c_col_offset + 1) * n + c_row_offset
        c10 = c_col_offset * n + (c_row_offset + 1)
        c11 = (c_col_offset + 1) * n + (c_row_offset + 1)
        
        # M1: (A11+A22)(B11+B22)
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2: (A11+A21)B11
        add_term(
            {a00: 1, a10: 1},
            {b00: 1},
            {c10: 1}
        )
        # M3: A11(B12-B22)
        add_term(
            {a00: 1},
            {b01: 1, b11: -1},
            {c01: 1}
        )
        # M4: A22(B21-B11)
        add_term(
            {a11: 1},
            {b10: 1, b00: -1},
            {c00: 1, c10: 1}
        )
        # M5: (A12+A22)B22
        add_term(
            {a01: 1, a11: 1},
            {b11: 1},
            {c01: 1, c11: -1}
        )
        # M6: (A11-A21)(B21+B22)
        add_term(
            {a00: 1, a10: -1},
            {b10: 1, b11: 1},
            {c11: 1}
        )
        # M7: (A12-A22)(B11+B21)
        add_term(
            {a01: 1, a11: -1},
            {b00: 1, b10: 1},
            {c00: 1}
        )
    
    def schoolbook_2x2_2x1(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Schoolbook for 2x2 * 2x1 sub-problem.
        A rows: a_row_offset, a_row_offset+1, cols 0,1
        B rows 0,1, col b_col_offset
        C rows: c_row_offset, c_row_offset+1, col c_col_offset
        """
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    a_idx = (a_row_offset + i) * m + j
                    b_idx = j * p + (b_col_offset + k)
                    c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )
    
    def schoolbook_2x3_3x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Schoolbook for 2x3 * 3x3 sub-problem.
        A rows: a_row_offset, a_row_offset+1, cols 2,3,4
        B rows 2,3,4, cols b_col_offset to b_col_offset+2
        C rows: c_row_offset, c_row_offset+1, cols c_col_offset to c_col_offset+2
        """
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    a_idx = (a_row_offset + i) * m + (2 + j)
                    b_idx = (2 + j) * p + (b_col_offset + k)
                    c_idx = (c_col_offset + k) * n + (c_row_offset + i)
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )
    
    # Build the decomposition
    # 4 C-blocks:
    # C_tl: A_top(2x5) * B_left(5x3), a_row=0, b_col=0, c_row=0, c_col=0
    # C_tr: A_top(2x5) * B_right(5x3), a_row=0, b_col=3, c_row=0, c_col=3
    # C_bl: A_bot(2x5) * B_left(5x3), a_row=2, b_col=0, c_row=2, c_col=0
    # C_br: A_bot(2x5) * B_right(5x3), a_row=2, b_col=3, c_row=2, c_col=3
    
    for a_row in [0, 2]:
        for b_col in [0, 3]:
            c_row = a_row
            c_col = b_col
            
            # Part 1: A_sub_L(2x2) * B_sub_T(2x3)
            # A_sub_L: rows a_row, a_row+1, cols 0,1
            # B_sub_T: rows 0,1, cols b_col to b_col+2
            # Split B_sub_T into BT_L(2x2) cols b_col,b_col+1 and BT_R(2x1) col b_col+2
            
            # Strassen on (2x2)(2x2): A_sub_L * BT_L
            strassen_2x2_2x2(a_row, b_col, c_row, c_col)
            
            # Schoolbook on (2x2)(2x1): A_sub_L * BT_R
            schoolbook_2x2_2x1(a_row, b_col + 2, c_row, c_col + 2)
            
            # Part 2: A_sub_R(2x3) * B_sub_B(3x3)
            # A_sub_R: rows a_row, a_row+1, cols 2,3,4
            # B_sub_B: rows 2,3,4, cols b_col to b_col+2
            schoolbook_2x3_3x3(a_row, b_col, c_row, c_col)
    
    # Build final matrices
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_list[c][i]
        for j in range(m * p):
            V[j][c] = V_list[c][j]
        for k in range(p * n):
            W[k][c] = W_list[c][k]
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return best_result

# EVOLVE-BLOCK-END
