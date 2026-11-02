import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, coeff_u, v_idx, coeff_v, w_idx, coeff_w):
        nonlocal col
        U[u_idx][col] = coeff_u
        V[v_idx][col] = coeff_v
        W[w_idx][col] = coeff_w
        col += 1

    def set_column(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] = val
        col += 1

    for b in range(6):
        u1 = {0: 1, 3: 1}
        v1 = {2*b: 1, 13 + 2*b + 1: 1}
        w1 = {4*b: 1, 4*b + 3: 1}
        set_column(u1, v1, w1)
        
        u2 = {2: 1, 3: 1}
        v2 = {2*b: 1}
        w2 = {4*b + 1: 1, 4*b + 3: -1}
        set_column(u2, v2, w2)
        
        u3 = {0: 1}
        v3 = {2*b + 1: 1, 13 + 2*b + 1: -1}
        w3 = {4*b + 2: 1, 4*b + 3: 1}
        set_column(u3, v3, w3)
        
        u4 = {3: 1}
        v4 = {13 + 2*b: 1, 2*b: -1}
        w4 = {4*b: 1, 4*b + 1: 1}
        set_column(u4, v4, w4)
        
        u5 = {0: 1, 1: 1}
        v5 = {13 + 2*b + 1: 1}
        w5 = {4*b: -1, 4*b + 2: 1}
        set_column(u5, v5, w5)
        
        u6 = {2: 1, 0: -1}
        v6 = {2*b: 1, 2*b + 1: 1}
        w6 = {4*b + 3: 1}
        set_column(u6, v6, w6)
        
        u7 = {1: 1, 3: -1}
        v7 = {13 + 2*b: 1, 13 + 2*b + 1: 1}
        w7 = {4*b: 1}
        set_column(u7, v7, w7)

    u_t1 = {0: 1}
    v_t1 = {12: 1}
    w_t1 = {24: 1}
    set_column(u_t1, v_t1, w_t1)
    
    u_t2 = {1: 1}
    v_t2 = {25: 1}
    w_t2 = {24: 1}
    set_column(u_t2, v_t2, w_t2)
    
    u_t3 = {2: 1}
    v_t3 = {12: 1}
    w_t3 = {25: 1}
    set_column(u_t3, v_t3, w_t3)
    
    u_t4 = {3: 1}
    v_t4 = {25: 1}
    w_t4 = {25: 1}
    set_column(u_t4, v_t4, w_t4)

    return {"U": U, "V": V, "W": W}
