import numpy as np

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_rank1(u_idx_vals, v_idx_vals, w_idx_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in u_idx_vals:
            U[idx][current_rank] += val
        for idx, val in v_idx_vals:
            V[idx][current_rank] += val
        for idx, val in w_idx_vals:
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        # Rank 8 decomposition for M(1,4,2)
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def add_2_2_2(i_off, j_off, k_off):
        # Rank 7 Strassen decomposition for M(2,2,2)
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # 1. (a00+a11)(b00+b11)
        add_rank1([(u(0,0), 1), (u(1,1), 1)], [(v(0,0), 1), (v(1,1), 1)], [(w(0,0), 1), (w(1,1), 1)])
        # 2. (a10+a11)(b00)
        add_rank1([(u(1,0), 1), (u(1,1), 1)], [(v(0,0), 1)], [(w(1,0), 1), (w(1,1), -1)])
        # 3. (a00)(b01-b11)
        add_rank1([(u(0,0), 1)], [(v(0,1), 1), (v(1,1), -1)], [(w(0,1), 1), (w(1,1), 1)])
        # 4. (a11)(b10-b00)
        add_rank1([(u(1,1), 1)], [(v(1,0), 1), (v(0,0), -1)], [(w(0,0), 1), (w(1,0), 1)])
        # 5. (a00+a01)(b11)
        add_rank1([(u(0,0), 1), (u(0,1), 1)], [(v(1,1), 1)], [(w(0,0), -1), (w(0,1), 1)])
        # 6. (a10-a00)(b00+b01)
        add_rank1([(u(1,0), 1), (u(0,0), -1)], [(v(0,0), 1), (v(0,1), 1)], [(w(1,1), 1)])
        # 7. (a01-a11)(b10+b11)
        add_rank1([(u(0,1), 1), (u(1,1), -1)], [(v(1,0), 1), (v(1,1), 1)], [(w(0,0), 1)])

    def add_2_4_2(i_off, j_off, k_off):
        # M(2,4,2) = M(2,2,2) + M(2,2,2)
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    # Decompose M(3,4,6) into 3 blocks of M(3,4,2)
    for block_k in range(3):
        k_off = block_k * 2
        # M(1,4,2) for row 0 of A
        add_1_4_2(0, 0, k_off)
        # M(2,4,2) for rows 1,2 of A
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}
