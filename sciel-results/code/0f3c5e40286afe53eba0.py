import time

def construct():
    n = 14
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute non-adjacency bitmasks
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask ^ adj[v] ^ (1 << v) # Exclude self and neighbors

    size = 1 << n
    
    # is_clique[mask] and is_ind[mask]
    # Use recurrence:
    # Let v be LSB of mask. rest = mask ^ (1<<v).
    # mask is clique iff rest is clique AND rest is subset of adj[v]
    # mask is ind iff rest is ind AND rest is subset of non_adj[v]
    
    is_clique = [False] * size
    is_ind = [False] * size
    is_good = [False] * size
    
    is_clique[0] = True
    is_ind[0] = True
    is_good[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and (rest & ~adj[v] == 0)
        is_ind[mask] = is_ind[rest] and (rest & ~non_adj[v] == 0)
        is_good[mask] = is_clique[mask] or is_ind[mask]

    # DP for minimum number of parts
    INF = float('inf')
    dp = [INF] * size
    parent = [-1] * size
    dp[0] = 0
    
    start_time = time.time()
    deadline = 50.0
    
    for mask in range(1, size):
        # Iterate submasks of mask
        # To optimize, we can iterate submasks that contain the lowest set bit of mask?
        # No, we need to check all valid submasks.
        # However, we can prune: if dp[mask] is already 1, we can't do better.
        # Also, we can break early if we find a submask such that dp[mask^sub] == 1, then dp[mask] becomes 2.
        
        sub = mask
        best_val = INF
        best_sub = -1
        
        while sub:
            if is_good[sub]:
                val = 1 + dp[mask ^ sub]
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2: # Can't be 1 unless mask itself is good, which is handled if sub==mask
                        # Actually if sub==mask, val=1+dp[0]=1. So min is 1.
                        # If we find val=2, we can't go lower than 1.
                        # But we might find val=1 later if sub==mask is good.
                        # So we must check sub==mask first or handle it.
                        pass
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        parent[mask] = best_sub
        
        if time.time() > deadline:
            # Should not happen for n=14
            break

    # Reconstruct
    if dp[full_mask] == INF:
        # Fallback
        return [[i] for i in range(n)]
        
    parts = []
    cur = full_mask
    while cur:
        sub = parent[cur]
        if sub == -1:
            break
        part = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m ^= lsb
        parts.append(part)
        cur ^= sub
        
    return parts

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
