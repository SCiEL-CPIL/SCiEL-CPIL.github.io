import math

def construct():
    P_data = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P_data['n']
    edges = P_data['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_ind
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for m in range(1, size):
        lsb = m & -m
        v_idx = lsb.bit_length() - 1
        rest = m ^ lsb
        
        # Check if adding v_idx to rest keeps it a clique
        # rest must be a clique, and v_idx must be connected to all in rest
        if is_clique[rest] and (rest & ~adj[v_idx] == 0):
            is_clique[m] = True
            
        # Check if adding v_idx to rest keeps it an independent set
        # rest must be an independent set, and v_idx must not be connected to any in rest
        if is_ind[rest] and (rest & adj[v_idx] == 0):
            is_ind[m] = True

    # Combine validity
    is_valid = [False] * size
    for i in range(size):
        is_valid[i] = is_clique[i] or is_ind[i]
        
    # DP for minimum set cover
    # dp[mask] = min parts to cover 'mask'
    # prev[mask] = the submask chosen in the last step
    INF = 1000
    dp = [INF] * size
    prev = [0] * size
    dp[0] = 0
    
    for mask in range(1, size):
        best_val = INF
        best_sub = 0
        # Iterate all non-empty submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                # If we pick 'sub' as one part, the rest is mask ^ sub
                val = dp[mask ^ sub] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        prev[mask] = best_sub
        
    # Reconstruct the partition
    parts = []
    cur = full_mask
    while cur:
        s = prev[cur]
        # Convert mask s to list of vertices
        part = []
        m = s
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            part.append(idx)
            m ^= lsb
        parts.append(part)
        cur = cur ^ s
        
    return parts
