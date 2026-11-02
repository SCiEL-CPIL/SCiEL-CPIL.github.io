import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U part: A indices (00, 01, 10, 11)
    # V part: B indices (00, 01, 10, 11)
    # W part: C indices (00, 01, 10, 11)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B10-B11) -> C01, C11
    # M4: A11(B01-B00) -> C00, C10
    # M5: (A00+A10)B11 -> C00(-1), C01
    # M6: (A10-A00)(B00+B10) -> C11
    # M7: (A11-A01)(B01+B11) -> C00, C11
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 1, 0], # M5: A00, A10
        [-1, 0, 1, 0],# M6: A00(-1), A10
        [0, -1, 0, 1] # M7: A01(-1), A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10, B11(-1)
        [-1, 1, 0, 0],# M4: B00(-1), B01
        [0, 0, 0, 1], # M5: B11
        [1, 0, 1, 0], # M6: B00, B10
        [0, 1, 0, 1]  # M7: B01, B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00(-1), C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    
    # Blocks:
    # 1: A cols 0-1, B rows 0-1, C cols 0-1
    # 2: A cols 0-1, B rows 0-1, C cols 2-3
    # 3: A cols 2-3, B rows 2-3, C cols 0-1
    # 4: A cols 2-3, B rows 2-3, C cols 2-3
    
    blocks = [
        (0, 0, 0), # a_col_offset, b_row_offset, c_col_offset
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    rank_offset = 0
    for a_off, b_off, c_off in blocks:
        for r in range(7):
            # U
            for a_local in range(4):
                val = strassen_U[r][a_local]
                if val != 0:
                    a_i = a_local // 2
                    a_j = a_local % 2
                    global_a_idx = a_i * m + (a_j + a_off)
                    U[global_a_idx][rank_offset + r] += val
                    
            # V
            for b_local in range(4):
                val = strassen_V[r][b_local]
                if val != 0:
                    b_j = b_local // 2
                    b_k = b_local % 2
                    global_b_idx = (b_j + b_off) * p + b_k
                    V[global_b_idx][rank_offset + r] += val
                    
            # W
            for c_local in range(4):
                val = strassen_W[r][c_local]
                if val != 0:
                    c_i = c_local // 2
                    c_k = c_local % 2
                    global_c_idx = (c_k + c_off) * n + c_i
                    W[global_c_idx][rank_offset + r] += val
                    
        rank_offset += 7
        
    return {"U": U, "V": V, "W": W}
