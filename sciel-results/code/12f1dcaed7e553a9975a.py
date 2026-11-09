def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Block 1: A_L, B_LL -> C_LL
    a1 = [idx_u(0,0), idx_u(0,1), idx_u(1,0), idx_u(1,1)]
    b1 = [idx_v(0,0), idx_v(0,1), idx_v(1,0), idx_v(1,1)]
    c1 = [idx_w(0,0), idx_w(1,0), idx_w(0,1), idx_w(1,1)] # C11, C12, C21, C22
    
    # Block 2: A_R, B_RL -> C_RL (same C indices as Block 1)
    a2 = [idx_u(0,2), idx_u(0,3), idx_u(1,2), idx_u(1,3)]
    b2 = [idx_v(2,0), idx_v(2,1), idx_v(3,0), idx_v(3,1)]
    c2 = c1
    
    # Block 3: A_L, B_LR -> C_LR
    a3 = a1
    b3 = [idx_v(0,2), idx_v(0,3), idx_v(1,2), idx_v(1,3)]
    c3 = [idx_w(2,0), idx_w(3,0), idx_w(2,1), idx_w(3,1)]
    
    # Block 4: A_R, B_RR -> C_RR (same C indices as Block 3)
    a4 = a2
    b4 = [idx_v(2,2), idx_v(2,3), idx_v(3,2), idx_v(3,3)]
    c4 = c3
    
    col = 0
    col = add_strassen_block(U, V, W, a1, b1, c1, col)
    col = add_strassen_block(U, V, W, a2, b2, c2, col)
    col = add_strassen_block(U, V, W, a3, b3, c3, col)
    col = add_strassen_block(U, V, W, a4, b4, c4, col)
    
    return {"U": U, "V": V, "W": W}
