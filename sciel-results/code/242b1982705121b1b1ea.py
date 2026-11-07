"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    total_masks = 1 << n
    is_clique = [False] * total_masks
    is_ind = [False] * total_masks
    is_homo = [False] * total_masks
    
    is_clique[0] = True
    is_ind[0] = True
    is_homo[0] = True
    
    # Precompute properties for all masks
    for mask in range(1, total_masks):
        v_bit = mask & -mask
        v_idx = v_bit.bit_length() - 1
        rest = mask ^ v_bit
        
        # Check if rest is clique and v is connected to all in rest
        if is_clique[rest] and (adj[v_idx] & rest) == rest:
            is_clique[mask] = True
            
        # Check if rest is independent and v is not connected to any in rest
        if is_ind[rest] and (adj[v_idx] & rest) == 0:
            is_ind[mask] = True
            
        is_homo[mask] = is_clique[mask] or is_ind[mask]
        
        # Time check (rarely needed for n=14)
        if time.time() > deadline:
             # Fallback to greedy if precomputation takes too long (shouldn't happen)
             # But for safety, we can break and use a simpler method, 
             # however n=14 precomputation is instant.
             pass

    # DP to find min number of parts
    # dp[mask] = min parts to cover mask
    # prev[mask] = the subset S chosen in the optimal step for mask
    INF = float('inf')
    dp = [INF] * total_masks
    prev = [0] * total_masks
    dp[0] = 0
    
    for mask in range(1, total_masks):
        v_bit = mask & -mask
        rest = mask ^ v_bit
        
        # Iterate through all submasks of 'rest'
        # S = sub | v_bit
        sub = rest
        while True:
            S = sub | v_bit
            if is_homo[S]:
                rem = mask ^ S
                val = dp[rem] + 1
                if val < dp[mask]:
                    dp[mask] = val
                    prev[mask] = S
            
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        # Time check
        if time.time() > deadline:
             # If we run out of time, we might not have the optimal solution for all masks
             # But n=14 is small enough that this loop finishes in milliseconds.
             pass

    # Reconstruct the partition
    mask = total_masks - 1
    parts = []
    while mask:
        S = prev[mask]
        # Extract vertices
        part = []
        m = S
        while m:
            b = m & -m
            idx = b.bit_length() - 1
            part.append(idx)
            m ^= b
        parts.append(part)
        mask ^= S
        
    # Ensure types are correct
    # parts is list of lists of ints
    return parts

# EVOLVE-BLOCK-END
