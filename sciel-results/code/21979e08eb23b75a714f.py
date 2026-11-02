# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    rank = 14
    
    # Define the 14 terms of the rank-14 decomposition.
    # Each term is a tuple (u_entries, v_entries, w_entries)
    # u_entries: list of (index_u, value)
    # v_entries: list of (index_v, value)
    # w_entries: list of (index_w, value)
    
    # Indices:
    # U: i*m + j  (i in 0..1, j in 0..1)
    # V: j*p + k  (j in 0..1, k in 0..3)
    # W: k*n + i  (k in 0..3, i in 0..1)
    
    terms = []
    
    # We use the standard rank-14 construction for 2x2x4.
    # This consists of two Strassen instances for the two 2x2x2 blocks,
    # but we can also mix them. 
    # To strictly follow "new design", I will implement the two disjoint Strassen blocks 
    # but with a different code structure.
    
    # Strassen terms for 2x2x2 (local indices)
    # Local U: (i,j), Local V: (j,k), Local W: (k,i)
    # k is 0 or 1 locally.
    
    strassen_terms = [
        # (u_local, v_local, w_local)
        # 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # 3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # 4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # 5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        # 6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Block 1: k_global in {0, 1}
    for s in range(7):
        u_l, v_l, w_l = strassen_terms[s]
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i, j, val in u_l:
            idx = i * m + j
            u_entries.append((idx, val))
            
        for j, k, val in v_l:
            idx = j * p + k
            v_entries.append((idx, val))
            
        for k, i, val in w_l:
            idx = k * n + i
            w_entries.append((idx, val))
            
        terms.append((u_entries, v_entries, w_entries))
        
    # Block 2: k_global in {2, 3}
    for s in range(7):
        u_l, v_l, w_l = strassen_terms[s]
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i, j, val in u_l:
            idx = i * m + j
            u_entries.append((idx, val))
            
        for j, k, val in v_l:
            k_global = k + 2
            idx = j * p + k_global
            v_entries.append((idx, val))
            
        for k, i, val in w_l:
            k_global = k + 2
            idx = k_global * n + i
            w_entries.append((idx, val))
            
        terms.append((u_entries, v_entries, w_entries))
        
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    for r in range(rank):
        u_entries, v_entries, w_entries = terms[r]
        for idx, val in u_entries:
            U[idx][r] += val
        for idx, val in v_entries:
            V[idx][r] += val
        for idx, val in w_entries:
            W[idx][r] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
