import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # L: A coeffs [a, b, c, d]
    # R: B coeffs [e, f, g, h]
    # D: C coeffs [C00, C01, C10, C11]
    
    strassen_L = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [1, 0, -1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    strassen_R = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    strassen_D = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1],# M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0],# M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Mapping for Block 0 (k=0,1)
    # R indices: [0, 1, 4, 5] corresponding to B00, B01, B10, B11
    # D indices: [0, 2, 1, 3] corresponding to C00, C01, C10, C11
    # Note: D vector order is [C00, C01, C10, C11]
    # Global W indices:
    # C00 -> 0
    # C01 -> 2
    # C10 -> 1
    # C11 -> 3
    
    r_map_0 = [0, 1, 4, 5]
    d_map_0 = [0, 2, 1, 3]
    
    # Mapping for Block 1 (k=2,3)
    # R indices: [2, 3, 6, 7] corresponding to B02, B03, B12, B13
    # D indices: [4, 6, 5, 7] corresponding to C02, C03, C12, C13
    # Global W indices:
    # C02 (row 0, col 2) -> 2*2 + 0 = 4
    # C03 (row 0, col 3) -> 3*2 + 0 = 6
    # C12 (row 1, col 2) -> 2*2 + 1 = 5
    # C13 (row 1, col 3) -> 3*2 + 1 = 7
    
    r_map_1 = [2, 3, 6, 7]
    d_map_1 = [4, 6, 5, 7]
    
    # Fill U, V, W
    for r in range(7):
        col_0 = r
        col_1 = 7 + r
        
        # Block 0
        for i in range(4):
            U[i][col_0] = strassen_L[r][i]
            V[r_map_0[i]][col_0] = strassen_R[r][i]
            W[d_map_0[i]][col_0] = strassen_D[r][i]
            
        # Block 1
        for i in range(4):
            U[i][col_1] = strassen_L[r][i]
            V[r_map_1[i]][col_1] = strassen_R[r][i]
            W[d_map_1[i]][col_1] = strassen_D[r][i]
            
    return {"U": U, "V": V, "W": W}
