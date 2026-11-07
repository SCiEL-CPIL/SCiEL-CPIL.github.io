import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid masks
    valid_masks = []
    for mask in range(1, 1 << n):
        is_c = True
        is_i = True
        # Check clique
        for u in range(n):
            if mask & (1 << u):
                others = mask & ~(1 << u)
                if others & ~adj[u]:
                    is_c = False
                    break
        if is_c:
            valid_masks.append(mask)
            continue
            
        # Check independent set
        for u in range(n):
            if mask & (1 << u):
                others = mask & ~(1 << u)
                if others & adj[u]:
                    is_i = False
                    break
        if is_i:
            valid_masks.append(mask)
            
    # DP
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    choice = [0] * (1 << n) # Store the mask chosen at this step
    
    # Optimization: sort valid_masks by size descending to find large parts earlier?
    # Doesn't matter for DP correctness, but might help if we add pruning.
    # For now, just iterate.
    
    for mask in range(1, 1 << n):
        best_val = INF
        best_part = 0
        for p in valid_masks:
            if p & mask == p:
                val = 1 + dp[mask ^ p]
                if val < best_val:
                    best_val = val
                    best_part = p
        dp[mask] = best_val
        choice[mask] = best_part
        
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        p = choice[mask]
        if p == 0:
            # Should not happen if solution exists
            break
        # Convert p to list of vertices
        part_verts = []
        for i in range(n):
            if p & (1 << i):
                part_verts.append(i)
        parts.append(part_verts)
        mask ^= p
        
    # Ensure types are correct
    result = []
    for part in parts:
        result.append([int(v) for v in part])
        
    return result
