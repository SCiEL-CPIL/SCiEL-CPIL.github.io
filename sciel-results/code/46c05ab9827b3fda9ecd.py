import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7-product scheme for 2x2 matrix multiplication
    # A[2x2], B[2x2], C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Four 2x2 sub-problems for 2x4 x 4x4 -> 2x4
    subs = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col_offset = 0
    for s in subs:
        a_r, a_c, b_r, b_c, c_r, c_c = s
        
        a_idx_map = [
            a_r * m + a_c,
            a_r * m + (a_c + 1),
            (a_r + 1) * m + a_c,
            (a_r + 1) * m + (a_c + 1)
        ]
        
        b_idx_map = [
            b_r * p + b_c,
            b_r * p + (b_c + 1),
            (b_r + 1) * p + b_c,
            (b_r + 1) * p + (b_c + 1)
        ]
        
        c_idx_map = [
            c_c * n + c_r,
            (c_c + 1) * n + c_r,
            c_c * n + (c_r + 1),
            (c_c + 1) * n + (c_r + 1)
        ]
        
        for k in range(7):
            col = col_offset + k
            for local_i, val in enumerate(strassen_U[k]):
                if val != 0:
                    U[a_idx_map[local_i]][col] = val
            for local_i, val in enumerate(strassen_V[k]):
                if val != 0:
                    V[b_idx_map[local_i]][col] = val
            for local_i, val in enumerate(strassen_W[k]):
                if val != 0:
                    W[c_idx_map[local_i]][col] = val
                    
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
