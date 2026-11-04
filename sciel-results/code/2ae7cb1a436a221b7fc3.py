import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0
    def add_term(u_rows, v_rows, w_rows, w_signs=None):
        nonlocal col
        if w_signs is None:
            w_signs = [1] * len(w_rows)
        for r in u_rows:
            U[r, col] = 1
        for r in v_rows:
            V[r, col] = 1
        for r, s in zip(w_rows, w_signs):
            W[r, col] = s
        col += 1

    # 1. A_1 (2x2) * B_1 (2x2) -> C_1 (2x2) using Strassen (7 mults)
    # A_1: i,j in 0,1. B_1: j,k in 0,1. C_1: i,k in 0,1.
    
    # M1 = (A00+A11)(B00+B11)
    add_term([u_idx(0,0), u_idx(1,1)], [v_idx(0,0), v_idx(1,1)], 
             [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)], [1,1,1,1])
             
    # M2 = (A10+A11)B00
    add_term([u_idx(1,0), u_idx(1,1)], [v_idx(0,0)], 
             [w_idx(0,1), w_idx(1,1)], [1,-1])
             
    # M3 = A00(B01-B11)
    add_term([u_idx(0,0)], [v_idx(0,1), v_idx(1,1)], 
             [w_idx(0,1), w_idx(1,1)], [1,1])
             
    # M4 = A11(B10-B00)
    add_term([u_idx(1,1)], [v_idx(1,0), v_idx(0,0)], 
             [w_idx(1,0), w_idx(1,1)], [1,1])
             
    # M5 = (A00+A01)B11
    add_term([u_idx(0,0), u_idx(0,1)], [v_idx(1,1)], 
             [w_idx(0,0), w_idx(1,1)], [1,1])
             
    # M6 = (A10-A00)(B00+B01)
    add_term([u_idx(1,0), u_idx(0,0)], [v_idx(0,0), v_idx(0,1)], 
             [w_idx(1,1)], [1])
             
    # M7 = (A01-A11)(B10+B11)
    add_term([u_idx(0,1), u_idx(1,1)], [v_idx(1,0), v_idx(1,1)], 
             [w_idx(0,0), w_idx(1,0)], [-1,1])

    # 2. A_1 (2x2) * B_2 (2x2) -> C_2 (2x2) using Strassen (7 mults)
    # B_2: j,k in 0,1 but global k is 2,3.
    # v_idx(j, k_global)
    
    # M1' = (A00+A11)(B02+B13)
    add_term([u_idx(0,0), u_idx(1,1)], [v_idx(0,2), v_idx(1,3)], 
             [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)], [1,1,1,1])
             
    # M2' = (A10+A11)B02
    add_term([u_idx(1,0), u_idx(1,1)], [v_idx(0,2)], 
             [w_idx(2,1), w_idx(3,1)], [1,-1])
             
    # M3' = A00(B03-B13)
    add_term([u_idx(0,0)], [v_idx(0,3), v_idx(1,3)], 
             [w_idx(2,1), w_idx(3,1)], [1,1])
             
    # M4' = A11(B12-B02)
    add_term([u_idx(1,1)], [v_idx(1,2), v_idx(0,2)], 
             [w_idx(3,0), w_idx(3,1)], [1,1])
             
    # M5' = (A00+A01)B13
    add_term([u_idx(0,0), u_idx(0,1)], [v_idx(1,3)], 
             [w_idx(2,0), w_idx(3,1)], [1,1])
             
    # M6' = (A10-A00)(B02+B03)
    add_term([u_idx(1,0), u_idx(0,0)], [v_idx(0,2), v_idx(0,3)], 
             [w_idx(3,1)], [1])
             
    # M7' = (A01-A11)(B12+B13)
    add_term([u_idx(0,1), u_idx(1,1)], [v_idx(1,2), v_idx(1,3)], 
             [w_idx(2,0), w_idx(3,0)], [-1,1])

    # 3. A_2 (2x1) * B_3 (1x4) -> C_3 (2x4) (8 mults)
    # A_2: i in 0,1; j=2.
    # B_3: j=2; k in 0..3.
    for i in range(n):
        for k in range(p):
            add_term([u_idx(i, 2)], [v_idx(2, k)], [w_idx(k, i)], [1])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
