import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Dimensions
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    rank = 42
    
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    
    # Local Strassen vectors for 2x2x2
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_local = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_local = [
        [1, 0, 0, 1], # M1 (b00+b11)
        [1, 0, 0, 0], # M2 (b00)
        [0, 1, 0, -1],# M3 (b01-b11)
        [-1, 0, 1, 0],# M4 (b10-b00)
        [0, 0, 0, 1], # M5 (b11)
        [1, 1, 0, 0], # M6 (b00+b01)
        [0, 0, 1, 1]  # M7 (b10+b11)
    ]
    
    W_local = [
        [1, 0, 0, 1], # M1 (c00+c11)
        [0, 0, 1, -1],# M2 (c10-c11)
        [0, 1, 0, 1], # M3 (c01+c11)
        [1, 0, 1, 0], # M4 (c00+c10)
        [-1, 1, 0, 0],# M5 (-c00+c01)
        [0, 0, 0, 1], # M6 (c11)
        [1, 0, 0, 0]  # M7 (c00)
    ]
    
    # Mapping functions
    # Global A index for local (r, c) where r in 0..1, c in 0..1
    # Left block: A cols 0,1. Global col = c.
    # Right block: A cols 2,3. Global col = 2 + c.
    def get_A_idx_L(r, c):
        return r * m + c
        
    def get_A_idx_R(r, c):
        return r * m + (2 + c)

    # Global B index for local (r, c) where r in 0..1, c in 0..1
    # Top block: B rows 0,1. Global row = r.
    # Bottom block: B rows 2,3. Global row = 2 + r.
    # Global col = 2*col_block + c.
    def get_B_idx_T(r, c, cb):
        return r * p + (2 * cb + c)

    def get_B_idx_B(r, c, cb):
        return (2 + r) * p + (2 * cb + c)

    # Global C index for local (r, c) where r in 0..1, c in 0..1
    # Task convention: W[k*n+i, r] where k is col index (0..5), i is row index (0..1).
    # Global col = 2*col_block + c.
    # Global row = r.
    # Index = Global Col * n + Global Row
    def get_C_idx(r, c, cb):
        global_col = 2 * cb + c
        return global_col * n + r

    col_offset = 0
    
    # Iterate over 3 column blocks
    for cb in range(3):
        # Left block (A_L * B_T)
        for t in range(7):
            r_col = col_offset
            # U
            for lr in range(2):
                for lc in range(2):
                    val = U_local[t][lr * 2 + lc]
                    if val != 0:
                        ga = get_A_idx_L(lr, lc)
                        U[ga][r_col] += val
                        
            # V
            for lr in range(2):
                for lc in range(2):
                    val = V_local[t][lr * 2 + lc]
                    if val != 0:
                        gb = get_B_idx_T(lr, lc, cb)
                        V[gb][r_col] += val
                        
            # W
            for lr in range(2):
                for lc in range(2):
                    val = W_local[t][lr * 2 + lc]
                    if val != 0:
                        gc = get_C_idx(lr, lc, cb)
                        W[gc][r_col] += val
                        
            col_offset += 1
            
        # Right block (A_R * B_B)
        for t in range(7):
            r_col = col_offset
            # U
            for lr in range(2):
                for lc in range(2):
                    val = U_local[t][lr * 2 + lc]
                    if val != 0:
                        ga = get_A_idx_R(lr, lc)
                        U[ga][r_col] += val
                        
            # V
            for lr in range(2):
                for lc in range(2):
                    val = V_local[t][lr * 2 + lc]
                    if val != 0:
                        gb = get_B_idx_B(lr, lc, cb)
                        V[gb][r_col] += val
                        
            # W
            for lr in range(2):
                for lc in range(2):
                    val = W_local[t][lr * 2 + lc]
                    if val != 0:
                        gc = get_C_idx(lr, lc, cb)
                        W[gc][r_col] += val
                        
            col_offset += 1

    return {"U": U, "V": V, "W": W}
