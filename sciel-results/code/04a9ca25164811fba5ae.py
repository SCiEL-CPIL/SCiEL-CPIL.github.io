import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin

    # Build adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    size = 1 << n
    full_mask = size - 1
    
    # Precompute is_clique and is_independent for all masks
    # Using lists of booleans
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    # We can combine into is_homog
    is_homog = [False] * size
    is_homog[0] = True # Empty set is both, but we don't use it in partition
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        c = is_clique[rest] and ((adj_masks[v] & rest) == rest)
        # Check indep
        i = is_indep[rest] and ((adj_masks[v] & rest) == 0)
        
        is_clique[mask] = c
        is_indep[mask] = i
        is_homog[mask] = c or i

    # DP
    # dp[mask] = min parts
    # Initialize with popcount (max possible parts)
    # Or just inf
    dp = [0] * size
    parent = [0] * size
    
    # dp[0] = 0
    for mask in range(1, size):
        # Initial guess: mask size
        # But we can just use inf
        best = float('inf')
        best_sub = 0
        sub = mask
        while sub:
            if is_homog[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best
        parent[mask] = best_sub
        
        # Time check (expensive, so do it rarely or not at all for n=14)
        # if time.time() > deadline:
        #    break

    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        s = parent[mask]
        if s == 0:
            # Fallback
            # This shouldn't happen if DP is correct
            parts = [[i] for i in range(n)]
            return parts
        part = []
        # Extract bits
        m = s
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m ^= lsb
        parts.append(part)
        mask ^= s
        
    return parts
