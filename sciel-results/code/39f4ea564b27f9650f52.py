    # Correct mapping for W indices in Strassen block
    # local_k 0: C00 (i=0, k=0) -> w_idx 0
    # local_k 1: C01 (i=0, k=1) -> w_idx 2
    # local_k 2: C10 (i=1, k=0) -> w_idx 1
    # local_k 3: C11 (i=1, k=1) -> w_idx 3
    w_idx_map = [0, 2, 1, 3]
    
    for u_c, v_c, w_c in strassen_terms:
        # U mapping
        # local_j 0: A00 (i=0, j=0) -> u_idx 0
        # local_j 1: A01 (i=0, j=1) -> u_idx 1
        # local_j 2: A10 (i=1, j=0) -> u_idx 2
        # local_j 3: A11 (i=1, j=1) -> u_idx 3
        # u_idx = i*m + j. m=5.
        # A00: 0*5+0=0
        # A01: 0*5+1=1
        # A10: 1*5+0=5
        # A11: 1*5+1=6
        u_idx_map = [0, 1, 5, 6]
        
        # V mapping
        # local_k 0: B00 (j=0, k=0) -> v_idx 0
        # local_k 1: B01 (j=0, k=1) -> v_idx 1
        # local_k 2: B10 (j=1, k=0) -> v_idx 5
        # local_k 3: B11 (j=1, k=1) -> v_idx 6
        # v_idx = j*p + k. p=5.
        v_idx_map = [0, 1, 5, 6]
        
        for l in range(4):
            U[u_idx_map[l], col_offset] = u_c[l]
            V[v_idx_map[l], col_offset] = v_c[l]
            W[w_idx_map[l], col_offset] = w_c[l]
            
        col_offset += 1
