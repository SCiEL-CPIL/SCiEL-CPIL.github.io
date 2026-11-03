def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2
    # u, v, w are lists of 4 elements
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u index: i*2+j
    # v index: j*2+k
    # w index: k*2+i
    
    s_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    s_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    s_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, -1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    # Block indices for A: (bi, bj) where bi, bj in {0,1}
    # Block indices for B: (bj, bk) where bj in {0,1}, bk in {0,1,2}
    # Product for C block (bi, bk) involves A(bi, bj) and B(bj, bk)
    
    for bi in range(2):
        for bk in range(3):
            for bj in range(2):
                # Add 7 Strassen terms for A(bi, bj) * B(bj, bk)
                for s in range(7):
                    # U indices
                    # A block (bi, bj) is at rows 2*bi .. 2*bi+1, cols 2*bj .. 2*bj+1
                    # U row index: i * m + j = i * 4 + j
                    # Local i: 0,1. Global i: 2*bi + local_i
                    # Local j: 0,1. Global j: 2*bj + local_j
                    # U row: (2*bi + li) * 4 + (2*bj + lj)
                    
                    # V indices
                    # B block (bj, bk) is at rows 2*bj .. 2*bj+1, cols 2*bk .. 2*bk+1
                    # V row index: j * p + k = j * 6 + k
                    # Local j: 0,1. Global j: 2*bj + lj
                    # Local k: 0,1. Global k: 2*bk + lk
                    # V row: (2*bj + lj) * 6 + (2*bk + lk)
                    
                    # W indices
                    # C block (bi, bk) is at rows 2*bi .. 2*bi+1, cols 2*bk .. 2*bk+1
                    # W row index: k * n + i = k * 4 + i
                    # Local k: 0,1. Global k: 2*bk + lk
                    # Local i: 0,1. Global i: 2*bi + li
                    # W row: (2*bk + lk) * 4 + (2*bi + li)
                    
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                u_val = s_u[s][li * 2 + lj]
                                if u_val != 0:
                                    u_row = (2 * bi + li) * 4 + (2 * bj + lj)
                                    U[u_row][col] += u_val
                                
                                v_val = s_v[s][lj * 2 + lk]
                                if v_val != 0:
                                    v_row = (2 * bj + lj) * 6 + (2 * bk + lk)
                                    V[v_row][col] += v_val
                                    
                                w_val = s_w[s][lk * 2 + li]
                                if w_val != 0:
                                    w_row = (2 * bk + lk) * 4 + (2 * bi + li)
                                    W[w_row][col] += w_val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
