"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # Use lists for speed
    adj_list = adj
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest] and (adj_list[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set
        if is_independent[rest] and (adj_list[v] & rest) == 0:
            is_independent[mask] = True
            
    # Precompute valid masks
    valid = [False] * size
    for i in range(size):
        if is_clique[i] or is_independent[i]:
            valid[i] = True

    # DP
    # dp[mask] = min parts to cover mask
    # parent[mask] = submask chosen last
    INF = 10**9
    dp = [INF] * size
    parent = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Deadline check
        if time.time() > deadline:
            # Fallback: return singletons (valid but not optimal)
            return [[i] for i in range(n)]
            
        # Optimization: If the mask itself is valid, dp is 1
        if valid[mask]:
            dp[mask] = 1
            parent[mask] = mask
            continue
            
        # Iterate submasks
        # We only need to consider proper submasks? 
        # No, if valid[mask] was false, no submask equal to mask is valid.
        # So we can iterate all submasks.
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct
    full_mask = size - 1
    if dp[full_mask] == INF:
        # Should not happen, but fallback to singletons
        return [[i] for i in range(n)]
        
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        # Extract vertices from sub
        verts = []
        temp = sub
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
        parts.append(verts)
        mask ^= sub
        
    # Return as list of lists of ints
    # Ensure types are correct
    return parts
# EVOLVE-BLOCK-END
