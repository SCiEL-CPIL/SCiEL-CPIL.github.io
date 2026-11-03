import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -> k*n+i. k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3
    
    U_terms = [
        [1, 0, 0, 1], # T1
        [0, 0, 1, 1], # T2
        [1, 0, 0, 0], # T3
        [0, 0, 0, 1], # T4
        [1, 0, 1, 0], # T5
        [-1, 0, 1, 0], # T6
        [0, -1, 0, 1]  # T7
    ]
    
    V_terms = [
        [1, 0, 0, 1], # T1
        [1, 0, 0, 0], # T2
        [0, 0, 1, -1], # T3
        [-1, 1, 0, 0], # T4
        [0, 0, 0, 1], # T5
        [1, 0, 1, 0], # T6
        [0, 1, 0, 1]  # T7
    ]
    
    W_terms = [
        [1, 0, 0, 1], # T1
        [0, 1, 0, -1], # T2
        [0, 1, 1, 0], # T3
        [-1, 0, 1, 0], # T4
        [1, 0, 0, 1], # T5
        [0, -1, 0, 1], # T6
        [1, 0, 1, 0]  # T7
    ]
    
    for col in range(rank):
        if time.time() > deadline:
            break
            
        g = col // 14
        rem = col % 14
        side = rem // 7
        t = rem % 7
        
        u_vec = U_terms[t]
        v_vec = V_terms[t]
        w_vec = W_terms[t]
        
        # Map U
        for l_u in range(4):
            if u_vec[l_u] != 0:
                i_loc = l_u // 2
                j_loc = l_u % 2
                i_g = i_loc
                j_g = j_loc if side == 0 else j_loc + 2
                idx_u = i_g * m + j_g
                U[idx_u][col] += u_vec[l_u]
                
        # Map V
        for l_v in range(4):
            if v_vec[l_v] != 0:
                j_loc = l_v // 2
                k_loc = l_v % 2
                j_g = j_loc if side == 0 else j_loc + 2
                k_g = 2 * g + k_loc
                idx_v = j_g * p + k_g
                V[idx_v][col] += v_vec[l_v]
                
        # Map W
        for l_w in range(4):
            if w_vec[l_w] != 0:
                k_loc = l_w // 2
                i_loc = l_w % 2
                k_g = 2 * g + k_loc
                i_g = i_loc
                idx_w = k_g * n + i_g
                W[idx_w][col] += w_vec[l_w]
                
    return {"U": U, "V": V, "W": W}
