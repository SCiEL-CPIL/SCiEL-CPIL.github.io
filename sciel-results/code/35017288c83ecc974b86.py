import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, val in u_pairs:
            u_col[idx] = val
        for idx, val in v_pairs:
            v_col[idx] = val
        for idx, val in w_pairs:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_block(j1, j2, k1, k2):
        # Adds 7 rank steps for 2x2x2 block
        # m1 = (a00+a11)(b00+b11)
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
                     
        # m2 = (a10+a11)*b00
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
                     
        # m3 = a00*(b01-b11)
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
                     
        # m4 = a11*(b10-b00)
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
                     
        # m5 = (a00+a01)*b11
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
                     
        # m6 = (a10-a00)*(b00+b01)
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
                     
        # m7 = (a01-a11)*(b10+b11)
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_schoolbook_block_2x1x2(j, k1, k2):
        # Adds 4 rank steps for 2x1x2 block
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    # Define the blocks
    # j split: 0,1 | 2,3 | 4
    # k split: 0,1 | 2,3 | 4,5
    
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for j_range in j_groups:
        for k_range in k_groups:
            if len(j_range) == 2 and len(k_range) == 2:
                add_strassen_block(j_range[0], j_range[1], k_range[0], k_range[1])
            elif len(j_range) == 1 and len(k_range) == 2:
                add_schoolbook_block_2x1x2(j_range[0], k_range[0], k_range[1])
            else:
                # Fallback to schoolbook if needed, though not triggered here
                for j in j_range:
                    for k in k_range:
                        for i in range(n):
                            add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])

    # Convert from column lists to row-major matrices: shape (rows, rank)
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]

    return {"U": U, "V": V, "W": W}
