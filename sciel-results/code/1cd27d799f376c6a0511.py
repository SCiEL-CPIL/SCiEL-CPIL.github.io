DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # u: indices 0..3 (i*2+j)
    # v: local indices 0..3 (j*2+k) -> mapped to global
    # w: local indices 0..3 (k*2+i) -> mapped to global
    
    # Define terms as (u_dict, v_dict, w_dict)
    # v_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # w_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Note: w index in C is k*n + i.
    # Local k, i.
    # If k=0, i=0 -> w=0.
    # If k=0, i=1 -> w=1.
    # If k=1, i=0 -> w=2.
    # If k=1, i=1 -> w=3.
    # So w_local index = k*2 + i.
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})       # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global offsets
        # V index: j*p + k.
        # j=0, k=2b -> 2b
        # j=0, k=2b+1 -> 2b+1
        # j=1, k=2b -> 14+2b
        # j=1, k=2b+1 -> 14+2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # W index: k*n + i.
        # k=2b, i=0 -> 4b
        # k=2b, i=1 -> 4b+1
        # k=2b+1, i=0 -> 4b+2
        # k=2b+1, i=1 -> 4b+3
        w_offsets = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for u_idx, val in u_dict.items():
                U[u_idx][col] += val
            for v_l_idx, val in v_dict.items():
                V[v_offsets[v_l_idx]][col] += val
            for w_l_idx, val in w_dict.items():
                W[w_offsets[w_l_idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
