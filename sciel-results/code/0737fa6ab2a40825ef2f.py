import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Rank 24 decomposition using block structure
    # B is 3x6, split into 3 blocks of 3x2.
    # Each 3x2 block multiplication with 2x3 A is decomposed into:
    # 1. Outer product for j=0 (Rank 1)
    # 2. Strassen for j=1,2 (Rank 7)
    # Total 8 per block, 3 blocks -> Rank 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(r, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i * m + j][r] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][r] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][r] += val

    for l in range(3):
        base_r = l * 8
        
        # 1. Outer product term for j=0
        # A_{i,0} * B_{0, 2l+k'}
        u_coeffs = {(0, 0): 1, (1, 0): 1}
        v_coeffs = {(0, 2*l): 1, (0, 2*l+1): 1}
        w_coeffs = {
            (2*l, 0): 1, (2*l, 1): 1,
            (2*l+1, 0): 1, (2*l+1, 1): 1
        }
        add_term(base_r, u_coeffs, v_coeffs, w_coeffs)
        
        # 2. Strassen terms for j=1,2
        # A_sub = [[A01, A02], [A11, A12]]
        # B_sub = [[B1,2l, B1,2l+1], [B2,2l, B2,2l+1]]
        
        # M1 = (A01 + A11) * (B1,2l + B2,2l+1)
        u1 = {(0, 1): 1, (1, 1): 1}
        v1 = {(1, 2*l): 1, (2, 2*l+1): 1}
        w1 = {(2*l, 0): 1, (2*l+1, 1): 1}
        add_term(base_r + 1, u1, v1, w1)
        
        # M2 = (A11 + A12) * B1,2l
        u2 = {(1, 1): 1, (1, 2): 1}
        v2 = {(1, 2*l): 1}
        w2 = {(2*l, 1): 1, (2*l+1, 1): -1}
        add_term(base_r + 2, u2, v2, w2)
        
        # M3 = A01 * (B1,2l+1 - B2,2l+1)
        u3 = {(0, 1): 1}
        v3 = {(1, 2*l+1): 1, (2, 2*l+1): -1}
        w3 = {(2*l+1, 0): 1, (2*l+1, 1): 1}
        add_term(base_r + 3, u3, v3, w3)
        
        # M4 = A12 * (B2,2l - B1,2l)
        u4 = {(1, 2): 1}
        v4 = {(2, 2*l): 1, (1, 2*l): -1}
        w4 = {(2*l, 0): 1, (2*l, 1): 1}
        add_term(base_r + 4, u4, v4, w4)
        
        # M5 = (A01 + A02) * B2,2l+1
        u5 = {(0, 1): 1, (0, 2): 1}
        v5 = {(2, 2*l+1): 1}
        w5 = {(2*l, 0): -1, (2*l+1, 0): 1}
        add_term(base_r + 5, u5, v5, w5)
        
        # M6 = (A11 - A01) * (B1,2l + B1,2l+1)
        u6 = {(1, 1): 1, (0, 1): -1}
        v6 = {(1, 2*l): 1, (1, 2*l+1): 1}
        w6 = {(2*l+1, 1): 1}
        add_term(base_r + 6, u6, v6, w6)
        
        # M7 = (A02 - A12) * (B2,2l + B2,2l+1)
        u7 = {(0, 2): 1, (1, 2): -1}
        v7 = {(2, 2*l): 1, (2, 2*l+1): 1}
        w7 = {(2*l, 0): 1}
        add_term(base_r + 7, u7, v7, w7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
