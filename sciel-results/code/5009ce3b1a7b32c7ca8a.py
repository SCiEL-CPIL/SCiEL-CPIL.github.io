import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
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
    
    # A: 00:0, 01:1, 02:2, 10:3, 11:4, 12:5
    # B: 00:0, 01:1, 02:2, 03:3, 10:4, 11:5, 12:6, 13:7, 20:8, 21:9, 22:10, 23:11
    # C: 00:0, 10:1, 20:2, 30:3, 01:4, 11:5, 21:6, 31:7
    
    # Strassen block 1: A[0:2,0:2] * B[0:2,0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (5, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (5, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(4, 1), (5, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (4, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(5, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Strassen block 2: A[0:2,0:2] * B[0:2,2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(2, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(6, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(3, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(2, 1), (6, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(2, -1), (3, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(2, 1)], 13)
    
    # Outer: A[:,2] * B[2,:]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(4, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(2, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(3, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(5, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(6, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U, "V": V, "W": W}
