import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    return _construct_66(n, m, p)

def _construct_66(n, m, p):
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
        
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
                
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # Standard Strassen-like decomposition for 2x2x2
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # M2 = (A10 + A11)B00 -> C10
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0)], [1])
        # M3 = A00(B01 - B11) -> C01
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1)], [1])
        # M4 = A10(B10 - B00) -> C10
        add_rank1([u(1,0)], [1], [v(1,0), v(0,0)], [1, -1], [w(1,0)], [1])
        # M5 = (A00 + A01)B11 -> C01, C11
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,1), w(1,1)], [1, 1])
        # M6 = (A01 - A10)(B10 + B11) -> C00, C11
        add_rank1([u(0,1), u(1,0)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # M7 = (A00 - A11)(B00 + B01) -> C00, C11
        add_rank1([u(0,0), u(1,1)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
        
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
        
    return {"U": U, "V": V, "W": W}
