import numpy as np
import time

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115.0  # 120s evaluator limit, keep margin

def construct():
    """
    Returns a rank-minimal tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    Use Strassen (rank 49) for A*B1 and A*B2.
    Use Schoolbook (rank 16) for A*B3.
    Total rank: 49 + 49 + 16 = 114.
    """
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: return a valid decomposition
        pass

    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2 terms: (u_coeffs, v_coeffs, w_coeffs)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        def u_idx(i, j):
            return i * m + j
        def v_idx(j, k):
            return j * p + k
        def w_idx(k, i):
            return k * n + i
        
        # 7 block-level Strassen multiplications for 4x4 * 4x4
        # Each: (A_side, B_side, C_side)
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: dict of (block_row, block_col) -> coeff
        block_mults = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            for u_meta, v_meta, w_meta in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for local_idx, meta_coeff in u_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, block_coeff in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[u_idx(gi, gj)] += meta_coeff * block_coeff
                
                for local_idx, meta_coeff in v_meta.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, block_coeff in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[v_idx(gj, gk)] += meta_coeff * block_coeff
                
                for local_idx, meta_coeff in w_meta.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[w_idx(gk, gi)] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)

    # A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # A * B3 (col 8) - Schoolbook
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
