"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute non-adjacency masks
    non_adj = [0] * n
    for i in range(n):
        non_adj[i] = full_mask ^ adj[i]
    
    # Precompute: for each mask, is it a clique or independent set?
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        is_clique = True
        is_independent = True
        
        # Check clique: for each v in mask, (mask without v) must be disjoint from non_adj[v]
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            remaining = mask ^ v_bit
            if remaining & non_adj[v]:
                is_clique = False
                break
            m ^= v_bit
        
        if not is_clique:
            # Check independent: for each v in mask, (mask without v) must be disjoint from adj[v]
            m = mask
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                remaining = mask ^ v_bit
                if remaining & adj[v]:
                    is_independent = False
                    break
                    m ^= v_bit
        
        valid[mask] = is_clique or is_independent
    
    # DP: dp[mask] = minimum number of valid parts to partition mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track which submask achieved the minimum for reconstruction
    parent_sub = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Iterate over all non-empty submasks of mask that are valid
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent_sub[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent_sub[mask]
        # Convert submask to list of vertices
        part = []
        s = sub
        while s:
            v_bit = s & (-s)
            v = v_bit.bit_length() - 1
            part.append(v)
            s ^= v_bit
        parts.append(part)
        mask = mask ^ sub
    
    # Ensure parts are lists of Python ints
    return parts

# EVOLVE-BLOCK-END
