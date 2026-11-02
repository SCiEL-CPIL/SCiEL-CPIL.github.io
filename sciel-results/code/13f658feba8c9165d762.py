def _build_rank18():
    """
    Rank-18 decomposition for 2x3x6.
    Composed of two Rank-9 decompositions for 2x3x3.
    Block 1: k in {0, 1, 2}
    Block 2: k in {3, 4, 5}
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2
    # Indices for local 2x2 block:
    # A: 0,1,2,3,4,5 (i*3+j) -> j in {0,1} -> 0,1,3,4
    # B: 0,1,2,3,4,5 (j*3+k) -> j in {0,1}, k in {0,1} -> 0,1,3,4
    # C: 0,1,2,3 (k*2+i) -> k in {0,1}
    
    strassen = [
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        ({4: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Residual term for j=2
    # U: i=0,j=2 -> 2; i=1,j=2 -> 5
    # V: j=2, k=0,1,2 -> 6,7,8
    # W: k=0,1,2; i=0,1 -> 0,1,2,3,4,5
    residual = ({2: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    
    block_terms = strassen + [residual]
    
    for block in range(2):
        for t_idx, (u_map, v_map, w_map) in enumerate(block_terms):
            col = block * 9 + t_idx
            for idx, val in u_map.items():
                U[idx, col] = val
            for l_idx, val in v_map.items():
                # l_idx is local j*3+k for 2x3x3
                # Global j is same. Global k is block*3 + local_k
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_k = block * 3 + k_local
                g_idx = j_local * p + g_k
                V[g_idx, col] = val
            for l_idx, val in w_map.items():
                # l_idx is local k*2+i for 2x3x3
                # Global k is block*3 + local_k. Global i is same.
                k_local = l_idx // 2
                i_local = l_idx % 2
                g_k = block * 3 + k_local
                g_idx = g_k * n + i_local
                W[g_idx, col] = val
                
    return {"U": U, "V": V, "W": W}
