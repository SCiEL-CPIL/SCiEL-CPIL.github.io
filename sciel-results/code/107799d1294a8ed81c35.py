import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 56
    
    # Strassen terms for 2x2 * 2x2:
    # (a_coeffs, b_coeffs, c_coeffs) with local indices 0=X11, 1=X12, 2=X21, 3=X22
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})              # M7
    ]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A indices (2x4, row-major: i*4+j)
            a11 = 0 * 4 + 2 * u
            a12 = 0 * 4 + 2 * u + 1
            a21 = 1 * 4 + 2 * u
            a22 = 1 * 4 + 2 * u + 1
            
            # B indices (4x8, row-major: j*8+k)
            b11 = (2 * u) * 8 + 2 * v
            b12 = (2 * u) * 8 + 2 * v + 1
            b21 = (2 * u + 1) * 8 + 2 * v
            b22 = (2 * u + 1) * 8 + 2 * v + 1
            
            # W indices (output: k*n+i = k*2+i)
            c11 = (2 * v) * 2 + 0
            c12 = (2 * v + 1) * 2 + 0
            c21 = (2 * v) * 2 + 1
            c22 = (2 * v + 1) * 2 + 1
            
            a_map = {0: a11, 1: a12, 2: a21, 3: a22}
            b_map = {0: b11, 1: b12, 2: b21, 3: b22}
            c_map = {0: c11, 1: c12, 2: c21, 3: c22}
            
            for s, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_terms):
                col = base_col + s
                for li, cf in a_coeffs.items():
                    U[a_map[li]][col] += cf
                for li, cf in b_coeffs.items():
                    V[b_map[li]][col] += cf
                for li, cf in c_coeffs.items():
                    W[c_map[li]][col] += cf
    
    return {"U": U, "V": V, "W": W}
