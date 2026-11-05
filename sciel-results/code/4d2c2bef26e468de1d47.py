import time

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # M(3,4,6) = M(1,4,6) + M(2,4,6)
    # M(1,4,6) rank 24
    # M(2,4,6) = M(2,2,6) + M(2,2,6)
    # M(2,2,6) = 3 * M(2,2,2) = 21
    # Total 24 + 2 * 21 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
        
    def add_1_4_6(i_off, j_off, k_off):
        # n=1, m=4, p=6
        for j in range(4):
            for k in range(6):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_2_6(i_off, j_off, k_off):
        # n=2, m=2, p=6
        # Split p into 2+2+2
        for k_block in range(3):
            add_2_2_2(i_off, j_off, k_off + k_block * 2)

    def add_2_4_6(i_off, j_off, k_off):
        # n=2, m=4, p=6
        # Split m into 2+2
        add_2_2_6(i_off, j_off, k_off)
        add_2_2_6(i_off, j_off + 2, k_off)

    # Build M(1,4,6) for n=0
    add_1_4_6(0, 0, 0)
    # Build M(2,4,6) for n=1,2
    add_2_4_6(1, 0, 0)

    return {"U": U, "V": V, "W": W}
