import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    size = 1 << n
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    # Precompute clique and independent set properties
    for mask in range(1, size):
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_ind[mask] = is_ind[rest] and ((adj[v] & rest) == 0)
        
    # DP to find minimum number of parts
    dp = [float('inf')] * size
    prev = [0] * size
    chosen = [0] * size
    dp[0] = 0
    
    for mask in range(1, size):
        sub = mask
        while sub:
            if is_clique[sub] or is_ind[sub]:
                prev_mask = mask ^ sub
                if dp[prev_mask] + 1 < dp[mask]:
                    dp[mask] = dp[prev_mask] + 1
                    prev[mask] = prev_mask
                    chosen[mask] = sub
            sub = (sub - 1) & mask
            
        # Time guard check (optional, but requested)
        # If we exceed time, we can't easily recover a valid partial solution from DP state
        # unless we store best valid partition found so far, which is not how DP works.
        # However, for n=14, this loop finishes in < 1s.
        if time.time() > deadline:
             # Fallback: return a trivial valid partition (each vertex separate)
             return [[i] for i in range(n)]

    # Reconstruct partition
    parts = []
    mask = size - 1
    while mask:
        sub = chosen[mask]
        part = [i for i in range(n) if (sub >> i) & 1]
        parts.append(part)
        mask = prev[mask]
        
    return parts
# EVOLVE-BLOCK-END
