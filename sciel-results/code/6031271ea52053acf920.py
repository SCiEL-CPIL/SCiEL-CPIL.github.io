import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank:
    # 3 * 7 (Strassen) = 21
    # 1 (Last Col B) = 1
    # 2 (Last Col A) = 2
    # Total = 24
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None: u_coeffs = [1] * len(u_indices)
        if v_coeffs is None: v_coeffs = [1] * len(v_indices)
        if w_coeffs is None: w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
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
        
        add_term([0, 4], [b00, b11], [c00, c11])
        add_term([3, 4], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([0], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([4], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([0, 1], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([3, 0], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([1, 4], [b10, b11], [c00], u_coeffs=[1, -1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Last Col B (1 term)
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    add_term([0, 1, 3, 4], [b06, b16], [c06, c16])
    
    # Last Col A (2 terms)
    # Term 1: U[0]=1, V[2]=1, W[even]=1
    # Term 2: U[1]=1, V[2]=1, W[odd]=1
    
    # V index for row 2 is 14..20
    # W indices for even C cols: 0, 2, 4, 6, 8, 10, 12
    # W indices for odd C cols: 1, 3, 5, 7, 9, 11, 13
    
    v_row2 = [14, 15, 16, 17, 18, 19, 20]
    w_even = [0, 2, 4, 6, 8, 10, 12]
    w_odd = [1, 3, 5, 7, 9, 11, 13]
    
    add_term([2], v_row2, w_even)
    add_term([5], v_row2, w_odd)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
