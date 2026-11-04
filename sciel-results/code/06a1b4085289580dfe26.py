import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # a_row, a_col: top-left of A block (2x2)
        # b_row, b_col: top-left of B block (2x2)
        # c_row, c_col: top-left of C block (2x2)
        
        # Local indices in A block:
        # p00: (a_row, a_col)
        # p01: (a_row, a_col+1)
        # p10: (a_row+1, a_col)
        # p11: (a_row+1, a_col+1)
        
        # Global A index: i * m + j
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        # Local indices in B block:
        # q00: (b_row, b_col)
        # q01: (b_row, b_col+1)
        # q10: (b_row+1, b_col)
        # q11: (b_row+1, b_col+1)
        
        # Global B index: j * p + k
        idx_q00 = b_row * p + b_col
        idx_q01 = b_row * p + (b_col + 1)
        idx_q10 = (b_row + 1) * p + b_col
        idx_q11 = (b_row + 1) * p + (b_col + 1)
        
        # Local indices in C block:
        # c00: (c_row, c_col)
        # c01: (c_row, c_col+1)
        # c10: (c_row+1, c_col)
        # c11: (c_row+1, c_col+1)
        
        # Global C index: k * n + i
        idx_c00 = c_col * n + c_row
        idx_c01 = (c_col + 1) * n + c_row
        idx_c10 = c_col * n + (c_row + 1)
        idx_c11 = (c_col + 1) * n + (c_row + 1)
        
        # Strassen M1..M7
        # M1: (p00+p11)(q00+q11) -> C00(+1), C11(+1)
        U[idx_p00][col] += 1
        U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M2: (p10+p11)q00 -> C10(+1), C11(-1)
        U[idx_p10][col] += 1
        U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        W[idx_c10][col] += 1
        W[idx_c11][col] -= 1
        col += 1
        
        # M3: p00(q01-q11) -> C01(+1), C11(+1)
        U[idx_p00][col] += 1
        V[idx_q01][col] += 1
        V[idx_q11][col] -= 1
        W[idx_c01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M4: p11(q10-q00) -> C00(+1), C10(+1)
        U[idx_p11][col] += 1
        V[idx_q10][col] += 1
        V[idx_q00][col] -= 1
        W[idx_c00][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        # M5: (p00+p01)q11 -> C00(-1), C01(+1)
        U[idx_p00][col] += 1
        U[idx_p01][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        # M6: (p10-p00)(q00+q01) -> C11(+1)
        U[idx_p10][col] += 1
        U[idx_p00][col] -= 1
        V[idx_q00][col] += 1
        V[idx_q01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M7: (p01-p11)(q10+q11) -> C00(+1)
        U[idx_p01][col] += 1
        U[idx_p11][col] -= 1
        V[idx_q10][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        col += 1

    # Helper to add a Schoolbook 2x2x1 block
    def add_schoolbook_221(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = p00 q0 + p01 q1
        # C1 = p10 q0 + p11 q1
        
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        idx_q0 = b_row * p + b_col
        idx_q1 = (b_row + 1) * p + b_col
        
        idx_c0 = c_col * n + c_row
        idx_c1 = c_col * n + (c_row + 1)
        
        # Term 1: p00 q0 -> C0
        U[idx_p00][col] += 1
        V[idx_q0][col] += 1
        W[idx_c0][col] += 1
        col += 1
        
        # Term 2: p01 q1 -> C0
        U[idx_p01][col] += 1
        V[idx_q1][col] += 1
        W[idx_c0][col] += 1
        col += 1
        
        # Term 3: p10 q0 -> C1
        U[idx_p10][col] += 1
        V[idx_q0][col] += 1
        W[idx_c1][col] += 1
        col += 1
        
        # Term 4: p11 q1 -> C1
        U[idx_p11][col] += 1
        V[idx_q1][col] += 1
        W[idx_c1][col] += 1
        col += 1

    # Part 1: 4 Strassen blocks for k=0..3
    # A_L is cols 0,1. A_R is cols 2,3.
    # B blocks are 2x2.
    
    # C block 0: k=0,1. Rows 0,1 of C.
    # C_00 (local) is C[0:2, 0:2] -> c_row=0, c_col=0
    # From A_L B_00 (b_row=0, b_col=0)
    add_strassen(0, 0, 0, 0, 0, 0)
    # From A_R B_10 (b_row=2, b_col=0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C block 1: k=2,3. Rows 0,1 of C.
    # C_01 (local) is C[0:2, 2:4] -> c_row=0, c_col=2
    # From A_L B_01 (b_row=0, b_col=2)
    add_strassen(0, 0, 0, 2, 0, 2)
    # From A_R B_11 (b_row=2, b_col=2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Part 2: 2 Schoolbook blocks for k=4
    # C block 2: k=4. Rows 0,1 of C.
    # c_row=0, c_col=4
    # From A_L B_L (b_row=0, b_col=4)
    add_schoolbook_221(0, 0, 0, 4, 0, 4)
    # From A_R B_R (b_row=2, b_col=4)
    add_schoolbook_221(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
