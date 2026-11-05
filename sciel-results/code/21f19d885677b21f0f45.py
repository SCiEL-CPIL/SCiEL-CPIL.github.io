import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Rank 60 decomposition
    # M(3,4,6) = 2 * M(3,4,3)
    # M(3,4,3) = M(1,4,3) rank 12 + M(2,4,3) rank 18
    # M(2,4,3) = 2 * M(2,2,3) rank 9
    # M(2,2,3) = M(2,2,2) rank 7 + M(2,2,1) rank 2
    
    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, current_rank] += val
        current_rank += 1
        
    def add_1_4_3(i_off, j_off, k_off):
        for j in range(4):
            for k in range(3):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
                
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # Strassen 7
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
        
    def add_2_2_1(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # M(2,2,1) rank 2
        # C_0 = A_00*B_00 + A_01*B_10
        # C_1 = A_10*B_00 + A_11*B_10
        # Term 1: (e_0+e_1) in U, e_0 in V, e_0 in W
        add_rank1([u(0,0), u(1,0)], [1, 1], [v(0,0)], [1], [w(0,0)], [1])
        # Term 2: (e_0+e_1) in U, e_1 in V, e_0 in W
        add_rank1([u(0,1), u(1,1)], [1, 1], [v(1,0)], [1], [w(0,0)], [1])
        
    def add_2_2_3(i_off, j_off, k_off):
        # M(2,2,3) = M(2,2,2) for k=0,1 + M(2,2,1) for k=2
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_1(i_off, j_off, k_off + 2)
        
    def add_2_4_3(i_off, j_off, k_off):
        add_2_2_3(i_off, j_off, k_off)
        add_2_2_3(i_off, j_off + 2, k_off)
        
    def add_3_4_3(i_off, j_off, k_off):
        add_1_4_3(i_off, j_off, k_off)
        add_2_4_3(i_off + 1, j_off, k_off)
        
    # Two blocks of 3 columns each
    add_3_4_3(0, 0, 0)
    add_3_4_3(0, 0, 3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
