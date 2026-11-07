import time

def construct():
    n = 14
    edges = P['edges']
    
    # Build adjacency bitmasks
    # adj[v] is a bitmask of neighbors of v
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_ind for all subsets
    # size = 2^n
    size = 1 << n
    is_clique = [False] * size
    is_ind = [False] * size
    is_good = [False] * size
    
    is_clique[0] = True
    is_ind[0] = True
    is_good[0] = True
    
    # We can compute is_clique[mask] based on submasks or direct check
    # Direct check for each mask:
    # mask is clique if for all v in mask, all other u in mask are in adj[v]
    # i.e., (mask & ~(adj[v] | (1<<v))) == 0
    
    for mask in range(1, size):
        # Check clique
        is_c = True
        # Iterate bits in mask
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Check if all other vertices in mask are neighbors of v
            # mask without v
            rest = mask ^ lsb
            # rest must be subset of adj[v]
            if rest & ~adj[v]:
                is_c = False
                break
            m ^= lsb
        is_clique[mask] = is_c
        
        # Check independent set
        is_i = True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            # rest must be subset of non-neighbors of v
            # non-neighbors of v are ~adj[v] & (size_mask - 1)
            # Let's precompute non_adj[v]
            # Actually, ~adj[v] in Python has infinite bits, so we mask it.
            non_adj_v = (~adj[v]) & (size - 1)
            if rest & ~non_adj_v:
                is_i = False
                break
            m ^= lsb
        is_ind[mask] = is_i
        
        is_good[mask] = is_c or is_i

    # DP
    INF = float('inf')
    dp = [INF] * size
    parent = [-1] * size
    dp[0] = 0
    
    full_mask = size - 1
    
    # Time guard
    start_time = time.time()
    deadline = 50.0 # 50 seconds, leaving 10s margin for 60s limit
    
    for mask in range(1, size):
        # Find lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Iterate submasks of mask that contain v
        # sub must be a submask of mask
        # and sub must contain lsb
        # We can iterate submasks of (mask ^ lsb) and then add lsb
        rest_mask = mask ^ lsb
        sub_rest = rest_mask
        best_val = INF
        best_sub = -1
        
        while True:
            sub = sub_rest | lsb
            if is_good[sub]:
                val = 1 + dp[mask ^ sub]
                if val < best_val:
                    best_val = val
                    best_sub = sub
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest_mask
            
        dp[mask] = best_val
        parent[mask] = best_sub
        
        if time.time() > deadline:
            # If we hit deadline, we should return best known valid solution
            # But DP is deterministic and fast, this shouldn't happen.
            # If it does, we break and try to reconstruct what we have?
            # No, if we break early, dp[full_mask] might be INF.
            # Given n=14, this loop is ~ 10^6 iterations max (due to lowest bit constraint reducing 3^n to ~ 2^n * n / 2 ?)
            # Actually, sum_{k=1}^n C(n,k) 2^{k-1} = (3^n - 1)/2.
            # 3^14 / 2 ~ 2.4 million iterations.
            # This is very fast in Python, likely < 1s.
            pass

    # Reconstruct
    if dp[full_mask] == INF:
        # Fallback: return a valid partition (e.g., singletons)
        # Singletons are always good (clique or ind set)
        return [[i] for i in range(n)]
        
    parts = []
    cur = full_mask
    while cur:
        sub = parent[cur]
        if sub == -1:
            # Should not happen if dp is valid
            break
        # Extract vertices from sub
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
