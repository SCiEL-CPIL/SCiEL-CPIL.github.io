def get_strassen_factors(n):
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    half = n // 2
    U_half, V_half, W_half = get_strassen_factors(half)
    rank_half = len(U_half[0])
    
    U = [[0]*(7*rank_half) for _ in range(n*n)]
    V = [[0]*(7*rank_half) for _ in range(n*n)]
    W = [[0]*(7*rank_half) for _ in range(n*n)]
    
    steps_def = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for s in range(7):
        A_b, B_b, C_b = steps_def[s]
        col_offset = s * rank_half
        
        for r in range(rank_half):
            # U
            for i in range(half):
                for j in range(half):
                    u_val = U_half[i*half+j][r]
                    if u_val == 0: continue
                    for br, bc, coeff in A_b:
                        gi = br*half + i
                        gj = bc*half + j
                        U[gi*n+gj][col_offset+r] += u_val * coeff
                        
            # V
            for j in range(half):
                for k in range(half):
                    v_val = V_half[j*half+k][r]
                    if v_val == 0: continue
                    for br, bc, coeff in B_b:
                        gj = br*half + j
                        gk = bc*half + k
                        V[gj*n+gk][col_offset+r] += v_val * coeff
                        
            # W
            for k in range(half):
                for i in range(half):
                    w_val = W_half[k*half+i][r]
                    if w_val == 0: continue
                    for br, bc, coeff in C_b:
                        gk = br*half + k
                        gi = bc*half + i
                        W[gk*n+gi][col_offset+r] += w_val * coeff
                        
    return U, V, W
