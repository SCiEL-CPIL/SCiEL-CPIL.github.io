import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_term(u_data, v_data, w_data):
        nonlocal col
        for idx, coef in u_data.items():
            U[idx][col] += coef
        for idx, coef in v_data.items():
            V[idx][col] += coef
        for idx, coef in w_data.items():
            W[idx][col] += coef
        col += 1
    
    def add_strassen_block(k_start):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        set_term({0: 1, 4: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        set_term({3: 1, 4: 1}, {b00: 1}, {c10: 1, c11: -1})
        set_term({0: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        set_term({4: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        set_term({0: 1, 1: 1}, {b11: 1}, {c00: -1, c01: 1})
        set_term({3: 1, 0: -1}, {b00: 1, b01: 1}, {c11: 1})
        set_term({1: 1, 4: -1}, {b10: 1, b11: 1}, {c00: 1})
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    set_term({0: 1}, {b06: 1}, {c06: 1})
    set_term({1: 1}, {b16: 1}, {c06: 1})
    set_term({3: 1}, {b06: 1}, {c16: 1})
    set_term({4: 1}, {b16: 1}, {c16: 1})
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        set_term({2: 1}, {b2k: 1}, {c0k: 1})
        set_term({5: 1}, {b2k: 1}, {c1k: 1})
    
    return {"U": U, "V": V, "W": W}
