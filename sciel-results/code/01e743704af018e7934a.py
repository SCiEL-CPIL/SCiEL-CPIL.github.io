import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start = time.time()
    deadline = start + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 50 decomposition
    rank = 50
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    def _elapsed_ok():
        return (time.time() - start) < deadline

    def add_strassen_block(u_idx, v_idx, w_idx, col):
        # u_idx: [a00, a01, a10, a11]
        # v_idx: [b00, b01, b10, b11]
        # w_idx: [c00, c01, c10, c11]
        
        # M1 = (a00+a11)(b00+b11)
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1  # c00
        W[w_idx[3]][col] += 1  # c11
        col += 1
        
        # M2 = (a10+a11)b00
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1  # c10
        col += 1
        
        # M3 = a00(b01-b11)
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1  # c01
        W[w_idx[3]][col] += 1  # c11
        col += 1
        
        # M4 = a11(b10-a10)
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        U[u_idx[2]][col] -= 1 # Wait, M4 = d(g-c). d is a11. g is b10. c is a10.
        # So U has a11 (+1) and a10 (-1).
        # V has b10 (+1).
        W[w_idx[0]][col] += 1  # c00
        W[w_idx[2]][col] += 1  # c10
        col += 1
        
        # M5 = (a00+a01)b11
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1  # c00
        W[w_idx[1]][col] += 1  # c01
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] -= 1
        V[v_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1  # c11
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] -= 1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1  # c00
        col += 1
        
        return col

    def add_schoolbook_2x2x1_block(u_idx, v_idx, w_idx, col):
        # u_idx: [a00, a01, a10, a11]
        # v_idx: [b00, b10] (b01, b11 are 0/unused)
        # w_idx: [c00, c10]
        
        # M1 = a00 b00
        U[u_idx[0]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # M2 = a01 b10
        U[u_idx[1]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # M3 = a10 b00
        U[u_idx[2]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # M4 = a11 b10
        U[u_idx[3]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        return col

    # Block 1: A_L B_T
    # A_L U-indices: [0, 1, 4, 5]
    u_L = [0, 1, 4, 5]
    
    # Sub-block 1.1: B_T cols 0,1
    # V-indices: [0, 1, 7, 8]
    v_T0 = [0, 1, 7, 8]
    # W-indices: [0, 1, 2, 3]
    w_C0 = [0, 1, 2, 3]
    col = add_strassen_block(u_L, v_T0, w_C0, 0)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 1.2: B_T cols 2,3
    # V-indices: [2, 3, 8, 9]
    v_T1 = [2, 3, 8, 9]
    # W-indices: [4, 5, 6, 7]
    w_C1 = [4, 5, 6, 7]
    col = add_strassen_block(u_L, v_T1, w_C1, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 1.3: B_T cols 4,5
    # V-indices: [4, 5, 10, 11]
    v_T2 = [4, 5, 10, 11]
    # W-indices: [8, 9, 10, 11]
    w_C2 = [8, 9, 10, 11]
    col = add_strassen_block(u_L, v_T2, w_C2, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 1.4: B_T col 6
    # V-indices: [6, 13]
    v_T3 = [6, 13]
    # W-indices: [12, 13]
    w_C3 = [12, 13]
    col = add_schoolbook_2x2x1_block(u_L, v_T3, w_C3, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Block 2: A_R B_B
    # A_R U-indices: [2, 3, 6, 7]
    u_R = [2, 3, 6, 7]
    
    # Sub-block 2.1: B_B cols 0,1 (rows 2,3)
    # V-indices: [14, 15, 21, 22]
    v_B0 = [14, 15, 21, 22]
    # W-indices: [0, 1, 2, 3]
    w_C0 = [0, 1, 2, 3]
    col = add_strassen_block(u_R, v_B0, w_C0, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 2.2: B_B cols 2,3 (rows 2,3)
    # V-indices: [16, 17, 23, 24]
    v_B1 = [16, 17, 23, 24]
    # W-indices: [4, 5, 6, 7]
    w_C1 = [4, 5, 6, 7]
    col = add_strassen_block(u_R, v_B1, w_C1, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 2.3: B_B cols 4,5 (rows 2,3)
    # V-indices: [18, 19, 25, 26]
    v_B2 = [18, 19, 25, 26]
    # W-indices: [8, 9, 10, 11]
    w_C2 = [8, 9, 10, 11]
    col = add_strassen_block(u_R, v_B2, w_C2, col)
    
    if not _elapsed_ok(): return {"U": U, "V": V, "W": W}
    
    # Sub-block 2.4: B_B col 6 (rows 2,3)
    # V-indices: [20, 27]
    v_B3 = [20, 27]
    # W-indices: [12, 13]
    w_C3 = [12, 13]
    col = add_schoolbook_2x2x1_block(u_R, v_B3, w_C3, col)
    
    return {"U": U, "V": V, "W": W}
