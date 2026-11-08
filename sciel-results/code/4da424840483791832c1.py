import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A: a00=0, a01=1, a10=3, a11=4
    # B: b00=0, b01=1, b10=4, b11=5
    # C (W index k*n+i): c00=0, c01=2, c10=1, c11=3
    
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)   # M1
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)           # M2
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)           # M3
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)           # M4
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)           # M5
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)           # M6
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)           # M7
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # B: b02=2, b03=3, b12=6, b13=7
    # C: c02=4, c03=6, c12=5, c13=7
    
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)    # M1'
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)           # M2'
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)           # M3'
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)          # M4'
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)          # M5'
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)          # M6'
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)          # M7'
    
    # Block 3: Direct for A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # A: a02=2, a12=5
    # B: b20=8, b21=9, b22=10, b23=11
    # C: c00=0,c01=2,c02=4,c03=6,c10=1,c11=3,c12=5,c13=7
    
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    # Verify all entries are integers or half-integers
    for mat in (U, V, W):
        for row in mat:
            for val in row:
                if val != int(val) and val * 2 != int(val * 2):
                    raise ValueError(f"Non-half-integer entry: {val}")
    
    return {"U": U, "V": V, "W": W}
