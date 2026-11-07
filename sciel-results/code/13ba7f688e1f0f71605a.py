import sys

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute validity of all subsets
    # is_valid[mask] is True if mask induces a clique or an independent set
    total_masks = 1 << n
    is_clique = [False] * total_masks
    is_independent = [False] * total_masks
    is_valid = [False] * total_masks
    
    is_clique[0] = True
    is_independent[0] = True
    is_valid[0] = True
    
    for mask in range(1, total_masks):
        # Get lowest set bit
        lsb = mask & -mask
        v_idx = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        # (adj_mask[v_idx] & rest) == rest means v is connected to all vertices in rest
        if is_clique[rest] and (adj_mask[v_idx] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        # (adj_mask[v_idx] & rest) == 0 means v is not connected to any vertex in rest
        if is_independent[rest] and (adj_mask[v_idx] & rest) == 0:
            is_independent[mask] = True
            
        is_valid[mask] = is_clique[mask] or is_independent[mask]
        
    # DP
    # dp[mask] = min parts to partition mask
    # choice[mask] = the submask T chosen for the first part in the partition of mask
    INF = float('inf')
    dp = [INF] * total_masks
    choice = [0] * total_masks
    dp[0] = 0
    
    for mask in range(1, total_masks):
        # Iterate over all submasks T of mask
        # We want to find T such that T is valid and 1 + dp[mask ^ T] is minimized
        sub = mask
        best_val = INF
        best_sub = 0
        
        while sub:
            if is_valid[sub]:
                val = 1 + dp[mask ^ sub]
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 1:
                        break # Can't do better than 1 part
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        choice[mask] = best_sub
        
    # Reconstruct partition
    full_mask = total_masks - 1
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = choice[current_mask]
        # Extract vertices from sub
        part_vertices = []
        temp = sub
        while temp:
            lsb = temp & -temp
            v_idx = (lsb.bit_length() - 1)
            part_vertices.append(v_idx)
            temp ^= lsb
        parts.append(part_vertices)
        current_mask ^= sub
        
    # The parts are in reverse order of selection, which doesn't matter for the set,
    # but let's return them as a list of lists of ints.
    # Ensure types are int and list
    return parts
