import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        non_adj[i] = full_mask & ~(adj[i] | (1 << i))
        
    # Precompute clique/indep/valid as integer bitmasks
    size = 1 << n
    clique_mask = 1  # bit 0 set: empty set is a clique
    indep_mask = 1   # bit 0 set: empty set is independent
    valid_mask = 1   # bit 0 set: empty set is valid
    
    for i in range(n):
        bit = 1 << (1 << i)
        clique_mask |= bit
        indep_mask |= bit
        valid_mask |= bit
        
    for mask in range(2, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        rest_bit = 1 << rest
        
        if (clique_mask & rest_bit) and (adj[v] & rest) == rest:
            clique_mask |= (1 << mask)
            
        if (indep_mask & rest_bit) and (non_adj[v] & rest) == rest:
            indep_mask |= (1 << mask)
            
        if (clique_mask >> mask) & 1 or (indep_mask >> mask) & 1:
            valid_mask |= (1 << mask)
        
    # DP for minimum number of parts
    dp = [0] * size
    INF = 1000
    
    for S in range(1, size):
        best = INF
        T = S
        while T:
            if (valid_mask >> T) & 1:
                val = 1 + dp[S ^ T]
                if val < best:
                    best = val
            T = (T - 1) & S
        dp[S] = best
        
        if time.time() > deadline:
            break

    # Backtrack to find the partition
    S = full_mask
    parts = []
    while S:
        T = S
        found = False
        while T:
            if (valid_mask >> T) & 1 and dp[S] == 1 + dp[S ^ T]:
                part_vertices = []
                tt = T
                while tt:
                    lsb = tt & -tt
                    v = lsb.bit_length() - 1
                    part_vertices.append(v)
                    tt ^= lsb
                parts.append(part_vertices)
                S ^= T
                found = True
                break
            T = (T - 1) & S
        if not found:
            break
            
    return parts

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
