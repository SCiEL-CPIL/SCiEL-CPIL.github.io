import time

_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = 3, 4, 6
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

    def add_1_4_2(i_off, j_off, k_off):
        # n=1, m=4, p=2: trivial decomposition of rank 8
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        # Strassen decomposition for 2x2x2, rank 7
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        add_rank1([(i_off+0)*m+(j_off+0), (i_off+1)*m+(j_off+1)],
                  [1, 1],
                  [(j_off+0)*p+(k_off+0), (j_off+1)*p+(k_off+1)],
                  [1, 1],
                  [(k_off+0)*n+(i_off+0), (k_off+1)*n+(i_off+1)],
                  [1, 1])
        # M2 = (A21+A22)B11 -> contributes to C21, C22
        add_rank1([(i_off+1)*m+(j_off+0), (i_off+1)*m+(j_off+1)],
                  [1, 1],
                  [(j_off+0)*p+(k_off+0)],
                  [1],
                  [(k_off+0)*n+(i_off+1), (k_off+1)*n+(i_off+1)],
                  [1, -1])
        # M3 = A11(B21-B11) -> contributes to C11, C12
        add_rank1([(i_off+0)*m+(j_off+0)],
                  [1],
                  [(j_off+0)*p+(k_off+1), (j_off+1)*p+(k_off+1)],
                  [1, -1],
                  [(k_off+1)*n+(i_off+0), (k_off+1)*n+(i_off+1)],
                  [1, 1])
        # M4 = A22(B12-B21) -> contributes to C12, C21
        add_rank1([(i_off+1)*m+(j_off+1)],
                  [1],
                  [(j_off+1)*p+(k_off+0), (j_off+0)*p+(k_off+0)],
                  [1, -1],
                  [(k_off+0)*n+(i_off+0), (k_off+0)*n+(i_off+1)],
                  [1, 1])
        # M5 = (A11+A12)B22 -> contributes to C11, C12
        add_rank1([(i_off+0)*m+(j_off+0), (i_off+0)*m+(j_off+1)],
                  [1, 1],
                  [(j_off+1)*p+(k_off+1)],
                  [1],
                  [(k_off+0)*n+(i_off+0), (k_off+1)*n+(i_off+0)],
                  [-1, 1])
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        add_rank1([(i_off+1)*m+(j_off+0), (i_off+0)*m+(j_off+0)],
                  [1, -1],
                  [(j_off+0)*p+(k_off+0), (j_off+0)*p+(k_off+1)],
                  [1, 1],
                  [(k_off+1)*n+(i_off+1)],
                  [1])
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        add_rank1([(i_off+0)*m+(j_off+1), (i_off+1)*m+(j_off+1)],
                  [1, -1],
                  [(j_off+1)*p+(k_off+0), (j_off+1)*p+(k_off+1)],
                  [1, 1],
                  [(k_off+0)*n+(i_off+0)],
                  [1])

    # Decompose M(3,4,6) as M(1,4,6) + M(2,4,6)
    # M(1,4,6) = M(1,4,2) + M(1,4,2) + M(1,4,2) = 3 * 8 = 24
    add_1_4_2(0, 0, 0)
    add_1_4_2(0, 0, 2)
    add_1_4_2(0, 0, 4)
    
    # M(2,4,6) = M(2,2,6) + M(2,2,6)
    # M(2,2,6) = M(2,2,2) + M(2,2,2) + M(2,2,2) = 3 * 7 = 21
    # So M(2,4,6) = 2 * 21 = 42
    # Total = 24 + 42 = 66
    
    add_2_2_2(1, 0, 0)
    add_2_2_2(1, 0, 2)
    add_2_2_2(1, 0, 4)
    add_2_2_2(1, 2, 0)
    add_2_2_2(1, 2, 2)
    add_2_2_2(1, 2, 4)

    return {"U": U, "V": V, "W": W}
