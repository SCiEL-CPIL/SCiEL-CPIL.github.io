"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute valid subsets (clique or independent set)
    # is_valid[mask] is True if mask induces a clique or an independent set
    is_valid = [False] * (1 << n)
    is_valid[0] = True
    
    # We can optimize the check by iterating through masks and using properties
    # But for n=14, direct check is fine.
    # A set is a clique if for every v in set, all other vertices in set are neighbors of v.
    # A set is an independent set if for every v in set, no other vertex in set is a neighbor of v.
    
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        # Check if independent
        is_independent = True
        
        # Iterate through vertices in the mask
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            others = mask ^ low
            
            if others & ~adj[v]:
                is_clique = False
            if others & adj[v]:
                is_independent = False
                
            if not is_clique and not is_independent:
                break
            m ^= low
            
        if is_clique or is_independent:
            is_valid[mask] = True

    # DP for minimum number of parts
    INF = 10**9
    dp = [INF] * (1 << n)
    prev = [-1] * (1 << n)
    dp[0] = 0
    
    # To speed up, we can precompute the list of valid submasks for each mask?
    # Or just iterate submasks. 3^14 is ~4.7M iterations, which is fast in Python if simple.
    
    for mask in range(1, 1 << n):
        # Check deadline
        if time.time() > deadline:
            break
            
        sub = mask
        while sub:
            if is_valid[sub]:
                cost = dp[mask ^ sub] + 1
                if cost < dp[mask]:
                    dp[mask] = cost
                    prev[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        sub = prev[mask]
        if sub == -1:
            # Fallback: should not happen for valid input
            # Find any valid submask
            s = mask
            found = False
            while s:
                if is_valid[s]:
                    sub = s
                    found = True
                    break
                s = (s - 1) & mask
            if not found:
                break
        part = []
        m = sub
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            part.append(int(v))
            m ^= low
        partition.append(part)
        mask ^= sub
        
    # Ensure return type is list of lists of ints
    # The problem states: "sequences must be list or tuple" and "integers must be Python int"
    # Our construction uses standard Python lists and ints.
    
    return partition

# EVOLVE-BLOCK-END
