<<<<<<< SEARCH
    # 3 Standard blocks for J3={4}
    # J3 x K1: j=4, k in {0,1}
    for k in [0, 1]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # J3 x K2: j=4, k in {2,3}
    for k in [2, 3]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # J3 x K3: j=4, k in {4,5}
    for k in [4, 5]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
=======
    # J3={4} slice: use a single rank-1 term for all 12 entries
    # u[i,4] = 1 for i=0,1; v[4,k] = 1 for k=0..5; w[k,i] = 1 for all k,i
    for i in [0, 1]:
        U[i * m + 4, col] = 1
    for k in range(p):
        V[4 * p + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    col += 1
>>>>>>> REPLACE
<<<<<<< SEARCH
    rank = 54
=======
    rank = 43
>>>>>>> REPLACE
