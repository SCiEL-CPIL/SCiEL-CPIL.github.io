import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)
    
    col = 0
    
    def add_strassen_term(col, j_offset_A, j_offset_B, k_offset_C, X_c, Y_c, Z_c):
        for (xi, xa, cx) in X_c:
            U[xi * m + (j_offset_A + xa)][col] += cx
        for (yb, yk, cy) in Y_c:
            V[(j_offset_B + yb) * p + (k_offset_C + yk)][col] += cy
        for (zi, zk, cz) in Z_c:
            W[(k_offset_C + zk) * n + zi][col] += cz
    
    strassen_terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for k_block in range(4):
        k_offset_C = k_block * 2
        for a_block in range(2):
            j_offset_A = a_block * 2
            j_offset_B = a_block * 2
            for (X_c, Y_c, Z_c) in strassen_terms:
                add_strassen_term(col, j_offset_A, j_offset_B, k_offset_C, X_c, Y_c, Z_c)
                col += 1
    
    return {"U": U, "V": V, "W": W}
