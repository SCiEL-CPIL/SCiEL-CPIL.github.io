import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    n = P['n']
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each mask, is it a clique? is it an independent set?
    num_masks = 1 << n
    is_clique = [False] * num_masks
    is_independent = [False] * num_masks
    is_valid = [False] * num_masks  # Precompute valid status to save OR in DP loop
    
    is_clique[0] = True
    is_independent[0] = True
    is_valid[0] = False # Empty set is not a valid part (parts must be non-empty)
    
    for mask in range(1, num_masks):
        # Get lowest set bit
        v_bit = mask & (-mask)
        v = v_bit.bit_length() - 1
        rest = mask ^ v_bit
        
        # Check if mask is a clique: rest must be subset of adj[v], and rest must be a clique
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        # Check if mask is an independent set: rest must be subset of ~adj[v], and rest must be independent
        non_adj = full_mask ^ adj[v]
        if is_independent[rest] and (rest & ~non_adj) == 0:
            is_independent[mask] = True
            
        if is_clique[mask] or is_independent[mask]:
            is_valid[mask] = True

    # DP: dp[mask] = min number of parts to partition vertices in mask
    INF = 10**9
    dp = [INF] * num_masks
    dp[0] = 0
    # Track which sub was chosen for reconstruction
    best_sub = [0] * num_masks
    
    for mask in range(1, num_masks):
        # Check deadline
        if time.time() > deadline:
            # Fallback: return a valid but potentially suboptimal partition
            # This should rarely trigger given n=14
            break
            
        sub = mask
        best_val = INF
        best_s = 0
        while sub:
            if is_valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_s = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        best_sub[mask] = best_s

    # If deadline was hit, we might not have computed dp[full_mask]
    # But for n=14, it's virtually impossible to hit the deadline.
    # If it did happen, we need a fallback.
    if dp[full_mask] == INF:
        # Simple greedy fallback
        remaining = full_mask
        parts = []
        while remaining:
            # Find a valid subset of remaining
            # Try to find a clique or independent set
            # Just pick the first vertex and form a singleton (always valid)
            # Or try to extend.
            # For safety, let's just do singletons.
            v_bit = remaining & (-remaining)
            parts.append([v_bit.bit_length() - 1])
            remaining ^= v_bit
        return parts

    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = best_sub[mask]
        # Extract vertices from sub
        verts = []
        m = sub
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            verts.append(v)
            m ^= v_bit
        parts.append(verts)
        mask ^= sub
    
    # Convert to list of lists (required type)
    return parts
# EVOLVE-BLOCK-END
