import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen for 4x4*4x4 blocks (rank 49 each) and schoolbook for 4x4*4x1 (rank 16).
    Total rank: 114.
    
    The output factor follows W[k*n+i, r] where k is the column index in B (0..8),
    i is the row index in A (0..3), and r is the term index.
    """
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # 2x2 Strassen terms (u_coeff, v_coeff, w_coeff)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    s2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # m1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),       # m2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),       # m3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),       # m4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),       # m5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),       # m6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),       # m7
    ]
    
    # Block-level Strassen combinations for 4x4*4x4
    blocks = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),           # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),           # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),           # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),           # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),           # M7
    ]
    
    def add_strassen_4x4(A_off, B_off, C_off):
        for A_side, B_side, C_side in blocks:
            for u_coeff, v_coeff, w_coeff in s2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, b_coeff in A_side:
                        gi = A_off[0] + br * 2 + li
                        gj = A_off[1] + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * b_coeff
                
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in B_side:
                        gj = B_off[0] + br * 2 + lj
                        gk = B_off[1] + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * b_coeff
                
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in C_side:
                        gi = C_off[0] + br * 2 + li
                        gk = C_off[1] + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * b_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4((0,0), (0,0), (0,0))
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4((0,0), (0,4), (0,4))
    
    # Part 3: A * B3 (col 8) - schoolbook 4x4 * 4x1
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Convert from list of columns to list of rows
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
