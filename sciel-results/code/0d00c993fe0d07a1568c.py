def sign_edges(graph, budget):
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Map logical edge indices (0..m-1) to actual edge indices in the input list
    # Bottom edge k connects k and (k+1)%m
    # Top edge k connects m+k and m+(k+1)%m
    # Matching edge k connects k and m+k
    
    bot_map = [-1] * m
    top_map = [-1] * m
    match_map = [-1] * m

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Bottom edge
            if v == (u + 1) % m:
                bot_map[u] = i
            elif u == (v + 1) % m:
                bot_map[v] = i
        elif u >= m and v >= m:
            # Top edge
            u_loc = u - m
            v_loc = v - m
            if v_loc == (u_loc + 1) % m:
                top_map[u_loc] = i
            elif u_loc == (v_loc + 1) % m:
                top_map[v_loc] = i
        else:
            # Matching edge
            if u < m:
                match_map[u] = i
            else:
                match_map[u - m] = i

    # Fallback if mapping failed (should not happen for valid prism)
    if -1 in bot_map or -1 in top_map or -1 in match_map:
        # Default assignment assuming standard order
        for k in range(m):
            if bot_map[k] == -1: bot_map[k] = k
            if top_map[k] == -1: top_map[k] = m + k
            if match_map[k] == -1: match_map[k] = 2*m + k

    # Pre-compute B and T matrices for all 2^m sign patterns
    # B_list[mask] is the signed adjacency matrix of the bottom cycle
    # T_list[mask] is the signed adjacency matrix of the top cycle
    
    num_masks = 2 ** m
    B_list = [None] * num_masks
    T_list = [None] * num_masks
    sign_arrays = [None] * num_masks # Store the sign vector for bottom/top for quick access
    
    for mask in range(num_masks):
        b = np.array([1 if (mask >> j) & 1 else -1 for j in range(m)], dtype=np.int8)
        sign_arrays[mask] = b
        
        B = np.zeros((m, m), dtype=np.float64)
        T = np.zeros((m, m), dtype=np.float64)
        
        for k in range(m):
            u = k
            v = (k + 1) % m
            B[u, v] = b[k]
            B[v, u] = b[k]
            T[u, v] = b[k] # T is same structure, just different signs later? 
            # Wait, T signs are independent. We pre-compute T for the same mask?
            # No, we iterate mask_b and mask_t separately.
            # So we just need the matrix structure.
            # Actually, B and T depend on the signs.
            # So we need to generate B for mask_b and T for mask_t.
            # Since the structure is identical, we can just store the sign vectors
            # and construct the matrix on the fly, or pre-compute all 2^m matrices.
            # Pre-computing 2^m matrices of size m x m is feasible for m<=9.
            
        # Re-do pre-computation properly
        B_list[mask] = B
        T_list[mask] = T # This is wrong, T depends on mask_t, not mask_b.
        # We should just store the sign vectors and construct matrices in the loop,
        # OR pre-compute all matrices for all masks.
        
    # Let's pre-compute all matrices
    B_list = [None] * num_masks
    T_list = [None] * num_masks
    
    for mask in range(num_masks):
        b = np.array([1 if (mask >> j) & 1 else -1 for j in range(m)], dtype=np.int8)
        B = np.zeros((m, m), dtype=np.float64)
        for k in range(m):
            u = k
            v = (k + 1) % m
            B[u, v] = b[k]
            B[v, u] = b[k]
        B_list[mask] = B
        T_list[mask] = B.copy() # T has same structure, signs are determined by mask_t
        
    best_signs = None
    best_radius = float('inf')
    
    # Iterate over all pairs of masks
    for mask_b in range(num_masks):
        B = B_list[mask_b]
        b_signs = sign_arrays[mask_b]
        
        for mask_t in range(num_masks):
            if time.time() > deadline:
                if best_signs is not None:
                    return np.asarray(best_signs, dtype=np.int8)
                # Fallback
                b_def = np.ones(m, dtype=np.int8)
                t_def = -np.ones(m, dtype=np.int8)
                signs = np.ones(E, dtype=np.int8)
                for k in range(m):
                    signs[bot_map[k]] = b_def[k]
                    signs[top_map[k]] = t_def[k]
                    signs[match_map[k]] = 1
                return signs

            t_signs = sign_arrays[mask_t]
            
            # Check unbalanced: b != t
            # If b == t, the graph is balanced (with matching +1)
            if np.array_equal(b_signs, t_signs):
                continue
            
            T = T_list[mask_t]
            
            # Construct A = [[B, I], [I, T]]
            A = np.zeros((2*m, 2*m), dtype=np.float64)
            A[:m, :m] = B
            A[m:, m:] = T
            A[:m, m:] = np.eye(m)
            A[m:, :m] = np.eye(m)
            
            w = np.linalg.eigvalsh(A)
            r = float(np.max(np.abs(w)))
            
            if r < best_radius:
                best_radius = r
                signs = np.ones(E, dtype=np.int8)
                for k in range(m):
                    signs[bot_map[k]] = b_signs[k]
                    signs[top_map[k]] = t_signs[k]
                    signs[match_map[k]] = 1
                best_signs = signs.copy()
                
    return np.asarray(best_signs, dtype=np.int8)
