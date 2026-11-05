import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
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
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
