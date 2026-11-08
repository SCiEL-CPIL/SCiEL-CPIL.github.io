import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen multipliers for 2x2
    # M1: (a+d)(e+h)
    # M2: (c+d)(e)
    # M3: (a)(f-h)
    # M4: (d)(g-e)
    # M5: (a+b)(h)
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # Coefficients for A and B linear forms
    # A indices: 0,1,2,3 (for 2x2 block)
    # B indices: 0,1,2,3 (for 2x2 block)
    
    strassen_A = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # Coefficients for C reconstruction
    # C indices: 0,1,2,3 (for 2x2 block)
    # M1: c11, c22
    # M2: c21, c22(-)
    # M3: c12, c22
    # M4: c11, c21
    # M5: c11(-), c12
    # M6: c22
    # M7: c11
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Sub-multiplications
    # 1. A1 B11 -> C1 (cols 0,1)
    # A1 is A[:, 0:2]. B11 is B[0:2, 0:2].
    # 2. A2 B21 -> C1 (cols 0,1)
    # A2 is A[:, 2:4]. B21 is B[2:4, 0:2].
    # 3. A1 B12 -> C2 (cols 2,3)
    # A1 is A[:, 0:2]. B12 is B[0:2, 2:4].
    # 4. A2 B22 -> C2 (cols 2,3)
    # A2 is A[:, 2:4]. B22 is B[2:4, 2:4].
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        nonlocal col
        for s in range(7):
            # Update U
            for ai in range(2):
                for aj in range(2):
                    val = strassen_A[s][ai*2+aj]
                    if val != 0:
                        u_idx = (a_row_offset + ai) * m + (a_col_offset + aj)
                        U[u_idx][col] += val
            # Update V
            for bi in range(2):
                for bj in range(2):
                    val = strassen_B[s][bi*2+bj]
                    if val != 0:
                        v_idx = (b_row_offset + bi) * p + (b_col_offset + bj)
                        V[v_idx][col] += val
            # Update W
            for ci in range(2):
                for cj in range(2):
                    val = strassen_W[s][ci*2+cj]
                    if val != 0:
                        w_idx = (c_col_offset + cj) * n + ci
                        W[w_idx][col] += val
            col += 1
            
    # 1. A1 (0,0) B11 (0,0) -> C1 (0)
    add_strassen_block(0, 0, 0, 0, 0)
    # 2. A2 (0,2) B21 (2,0) -> C1 (0)
    add_strassen_block(0, 2, 2, 0, 0)
    # 3. A1 (0,0) B12 (0,2) -> C2 (2)
    add_strassen_block(0, 0, 0, 2, 2)
    # 4. A2 (0,2) B22 (2,2) -> C2 (2)
    add_strassen_block(0, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
