import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # rank = 28
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components definitions
    # Each entry: (A_coeffs, B_coeffs, C_coeffs)
    # A, B, C are 4-vectors corresponding to 2x2 matrices in row-major order
    # a, b, c, d -> 0, 1, 2, 3
    strassen_components = [
        # M1
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Block mappings
    # Block 0: A1 (cols 0,1), B11 (rows 0,1; cols 0,1), C left (cols 0,1)
    # A local -> global idx
    # 0->0, 1->1, 2->4, 3->5
    map_A_0 = [0, 1, 4, 5]
    # B local -> global idx
    # 0->0, 1->1, 2->4, 3->5
    map_B_0 = [0, 1, 4, 5]
    # C local -> global idx (W index k*n+i)
    # c00 (0,0) -> 0*2+0=0
    # c01 (0,1) -> 1*2+0=2
    # c10 (1,0) -> 0*2+1=1
    # c11 (1,1) -> 1*2+1=3
    map_C_0 = [0, 2, 1, 3]
    
    # Block 1: A2 (cols 2,3), B21 (rows 2,3; cols 0,1), C left (cols 0,1)
    map_A_1 = [2, 3, 6, 7]
    map_B_1 = [8, 9, 12, 13]
    map_C_1 = [0, 2, 1, 3]
    
    # Block 2: A1 (cols 0,1), B12 (rows 0,1; cols 2,3), C right (cols 2,3)
    map_A_2 = [0, 1, 4, 5]
    map_B_2 = [2, 3, 6, 7]
    # c00 (0,2) -> 2*2+0=4
    # c01 (0,3) -> 3*2+0=6
    # c10 (1,2) -> 2*2+1=5
    # c11 (1,3) -> 3*2+1=7
    map_C_2 = [4, 6, 5, 7]
    
    # Block 3: A2 (cols 2,3), B22 (rows 2,3; cols 2,3), C right (cols 2,3)
    map_A_3 = [2, 3, 6, 7]
    map_B_3 = [10, 11, 14, 15]
    map_C_3 = [4, 6, 5, 7]
    
    blocks = [
        (map_A_0, map_B_0, map_C_0),
        (map_A_1, map_B_1, map_C_1),
        (map_A_2, map_B_2, map_C_2),
        (map_A_3, map_B_3, map_C_3),
    ]
    
    col = 0
    for map_A, map_B, map_C in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_components:
            for l in range(4):
                U[map_A[l]][col] = a_coeffs[l]
                V[map_B[l]][col] = b_coeffs[l]
                W[map_C[l]][col] = c_coeffs[l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
