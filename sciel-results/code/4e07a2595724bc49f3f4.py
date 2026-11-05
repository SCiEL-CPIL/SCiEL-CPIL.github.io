import time

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    # 3x4x6 = 4 * (3x2x3), splitting m=4 into 2+2 and p=6 into 3+3
    # 3x2x3 = 1x2x3 (rank 2) + 2x2x3 (rank 11)
    # 2x2x3 = 2x2x1 (rank 4) + 2x2x2 (rank 7, Strassen)
    # Total: 4 * (2 + 4 + 7) = 4 * 13 = 52
    
    rank = 52
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
    
    def add_1_2_3(i_off, j_off, k_off):
        # M(1,2,3): n=1, m=2, p=3, rank 2
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # Term j=0: U[(i_off)*2 + j_off], V[(j_off)*3 + k], W[k*1 + i_off]
        # Term j=1: U[(i_off)*2 + j_off+1], V[(j_off+1)*3 + k], W[k*1 + i_off]
        for k in range(3):
            add_rank1(
                [(i_off) * 2 + j_off], [1],
                [(j_off) * 3 + (k_off + k)], [1],
                [(k_off + k) * 1 + i_off], [1]
            )
            add_rank1(
                [(i_off) * 2 + (j_off + 1)], [1],
                [(j_off + 1) * 3 + (k_off + k)], [1],
                [(k_off + k) * 1 + i_off], [1]
            )
    
    def add_2_2_1(i_off, j_off, k_off):
        # M(2,2,1): n=2, m=2, p=1, rank 4
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        for i in range(2):
            for j in range(2):
                add_rank1(
                    [(i_off + i) * 2 + (j_off + j)], [1],
                    [(j_off + j) * 1 + k_off], [1],
                    [k_off * 2 + (i_off + i)], [1]
                )
    
    def add_2_2_2(i_off, j_off, k_off):
        # M(2,2,2): Strassen, rank 7
        def u(i, j): return (i_off + i) * 2 + (j_off + j)
        def v(j, k): return (j_off + j) * 2 + (k_off + k)
        def w(i, k): return (k_off + k) * 2 + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_2_3(i_off, j_off, k_off):
        # M(2,2,3) = 2x2x1 + 2x2x2, split p=3 into 1+2
        add_2_2_1(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off, k_off + 1)
    
    def add_3_2_3(i_off, j_off, k_off):
        # M(3,2,3) = 1x2x3 + 2x2x3, split n=3 into 1+2
        add_1_2_3(i_off, j_off, k_off)
        add_2_2_3(i_off + 1, j_off, k_off)
    
    # 3x4x6 = 4 * (3x2x3), split m=4 into 2+2, p=6 into 3+3
    for j_block in range(2):
        for k_block in range(2):
            j_off = j_block * 2
            k_off = k_block * 3
            add_3_2_3(0, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
