import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Non-adjacency bitmask (excluding self)
    non_adj = [0] * n
    for u in range(n):
        # All vertices except u
        mask = (1 << n) - 1
        mask ^= (1 << u)
        # Non-adjacent are those in mask but not in adj[u]
        non_adj[u] = mask & ~adj[u]

    # Precompute validity for all subsets
    # valid[mask] = True if mask is a clique or independent set
    valid = [False] * (1 << n)
    for mask in range(1 << n):
        if mask == 0:
            valid[mask] = True
            continue
        # Check clique
        is_c = True
        # Check independent
        is_i = True
        
        # Optimization: just check pairs
        # A set is a clique if for every vertex v in mask, (mask without v) is subset of adj[v]
        # A set is independent if for every vertex v in mask, (mask without v) is subset of non_adj[v]
        
        m = mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            
            if (adj[v] & (mask ^ v_bit)) != (mask ^ v_bit):
                is_c = False
            if (non_adj[v] & (mask ^ v_bit)) != (mask ^ v_bit):
                is_i = False
                
            if not is_c and not is_i:
                break
        
        valid[mask] = is_c or is_i

    # Dynamic Programming to find minimum number of parts
    # dp[mask] = minimum number of parts needed to cover 'mask'
    # Transition: dp[mask] = 1 + min(dp[mask ^ submask]) for all valid submasks of mask
    # To optimize, we can iterate submasks.
    # Complexity: O(3^n). For n=14, 3^14 = 4,782,969. This is very fast.
    
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # We also need to reconstruct the solution
    # best_sub[mask] stores the valid submask chosen to minimize dp[mask]
    best_sub = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Iterate through all submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct the partition
    full_mask = (1 << n) - 1
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        # Convert submask to list of vertices
        verts = []
        m = sub
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            verts.append(v)
            m ^= v_bit
        parts.append(verts)
        current_mask ^= sub
        
    # Ensure types are correct (int, list)
    # The problem states: "Types are checked exactly: integers must be Python int (NumPy integers and bools are rejected) and sequences must be list or tuple"
    # Our construction uses Python ints and lists, so this is fine.
    
    return parts
