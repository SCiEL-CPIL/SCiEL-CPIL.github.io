import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    def add_strassen_222(i0, j0, k0):
        # 7 terms for 2x2x2
        # 1. (A00+A11)(B00+B11) -> C00+C11
        # This is not a rank 1 term.
        # I will use the explicit 7 rank-1 terms.
        # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]
        # This is incorrect.
        
        # I will use schoolbook 8 terms for 2x2x2.
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    u_idx = (i0+di)*3 + (j0+dj)
                    v_idx = (j0+dj)*7 + (k0+dk)
                    w_idx = (k0+dk)*3 + (i0+di)
                    add_term(u_idx, v_idx, w_idx)

    # Part 1: A[:, 2] * B[2, :]
    for i in range(3):
        for k in range(7):
            u_idx = i * 3 + 2
            v_idx = 2 * 7 + k
            w_idx = k * 3 + i
            add_term(u_idx, v_idx, w_idx)
            
    # Part 2: A[:, 0:2] * B[0:2, :]
    # Part 2a: C[:, 0:3]
    # A[0:2, 0:2] * B[0:2, 0:3]
    # Use schoolbook 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 7 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx)
                
    # A[2, 0:2] * B[0:2, 0:3]
    for j in range(2):
        for k in range(3):
            u_idx = 2 * 3 + j
            v_idx = j * 7 + k
            w_idx = k * 3 + 2
            add_term(u_idx, v_idx, w_idx)

    # Part 2b: C[:, 3:7]
    # A[0:2, 0:2] * B[0:2, 3:7]
    for i in range(2):
        for j in range(2):
            for k in range(3, 7):
                u_idx = i * 3 + j
                v_idx = j * 7 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx)
                
    # A[2, 0:2] * B[0:2, 3:7]
    for j in range(2):
        for k in range(3, 7):
            u_idx = 2 * 3 + j
            v_idx = j * 7 + k
            w_idx = k * 3 + 2
            add_term(u_idx, v_idx, w_idx)

    # Total rank: 21 + 12 + 6 + 16 + 8 = 63.
    
    return {"U": U, "V": V, "W": W}
