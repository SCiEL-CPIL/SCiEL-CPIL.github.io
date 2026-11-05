import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 118.0  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_term(i, k, j, cu=1.0, cv=1.0, cw=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + k] = cu
        v_vec[k * p + j] = cv
        w_vec[j * n + i] = cw
        add_rank1(u_vec, v_vec, w_vec)
    
    # Nested Strassen for 4x4x4: 7 outer products x 7 inner products = 49 terms
    # Outer Strassen (4x4 blocks of 2x2):
    outer_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},
    ]
    
    # Inner Strassen (2x2 blocks):
    inner_products = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    def a_global(block, li, lk):
        return (li + (block // 2) * 2, lk + (block % 2) * 2)
    
    def b_global(block, lk, lj):
        return (lk + (block // 2) * 2, lj + (block % 2) * 2)
    
    def c_global(block, li, lj):
        return (li + (block // 2) * 2, lj + (block % 2) * 2)
    
    # Generate 49 core terms
    for op in outer_products:
        for ip in inner_products:
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in op['A'].items():
                for li, lk, inner_sign in ip['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in op['B'].items():
                for lk, lj, inner_sign in ip['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in op['C'].items():
                for li, lj, inner_sign in ip['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Tail terms: j=4 (k=0..4) and k=4 (j=0..4), avoiding double-count of (i,4,4)
    # j=4: i in 0..3, k in 0..4
    for i in range(4):
        for k in range(5):
            if time.time() > _DEADLINE:
                break
            add_term(i, k, 4)
    # k=4: i in 0..3, j in 0..3 (j=4 already covered above)
    for i in range(4):
        for j in range(4):
            if time.time() > _DEADLINE:
                break
            add_term(i, 4, j)
    
    if not U_cols:
        add_term(0, 0, 0)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
