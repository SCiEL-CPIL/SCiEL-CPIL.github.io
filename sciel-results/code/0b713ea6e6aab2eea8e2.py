import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 55.0  # 55s limit
    
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_valid = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for S in range(1, 1 << n):
        u_bit = S & -S
        u = u_bit.bit_length() - 1
        S_rest = S ^ u_bit
        
        if is_clique[S_rest]:
            if (S_rest & ~adj[u]) == 0:
                is_clique[S] = True
                
        if is_indep[S_rest]:
            if (S_rest & adj[u]) == 0:
                is_indep[S] = True
                
        if is_clique[S] or is_indep[S]:
            is_valid[S] = True
            
    dp = [float('inf')] * (1 << n)
    parent = [0] * (1 << n)
    dp[0] = 0
    
    timed_out = False
    for S in range(1, 1 << n):
        if time.time() > deadline:
            timed_out = True
            break
            
        T = S
        best_val = float('inf')
        best_T = 0
        while T:
            if is_valid[T]:
                val = 1 + dp[S ^ T]
                if val < best_val:
                    best_val = val
                    best_T = T
            T = (T - 1) & S
            
        if best_T != 0:
            dp[S] = best_val
            parent[S] = best_T
            
    if timed_out:
        # Fallback to singletons
        return [[i] for i in range(n)]
        
    parts = []
    S = full_mask
    while S:
        T = parent[S]
        if T == 0:
            # Should not happen
            return [[i] for i in range(n)]
        verts = []
        temp = T
        while temp:
            bit = temp & -temp
            v = bit.bit_length() - 1
            verts.append(v)
            temp ^= bit
        parts.append(verts)
        S ^= T
        
    return parts
