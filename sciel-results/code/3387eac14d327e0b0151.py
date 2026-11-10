import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    size = 1 << n
    
    # Precompute is_clique and is_indep
    is_clique = [False] * size
    is_indep = [False] * size
    valid = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    valid[0] = True
    
    for S in range(1, size):
        lsb = S & -S
        v = lsb.bit_length() - 1
        S_prime = S ^ lsb
        
        if is_clique[S_prime]:
            if (adj[v] & S_prime) == S_prime:
                is_clique[S] = True
        # else remains False
        
        if is_indep[S_prime]:
            if (adj[v] & S_prime) == 0:
                is_indep[S] = True
        # else remains False
        
        valid[S] = is_clique[S] or is_indep[S]
        
    # Greedy fallback
    def greedy_partition():
        # Sort by degree descending
        degs = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -degs[v])
        parts = [] # list of (mask, type) type 1:clique, 0:indep, 2:free
        for v in order:
            bit = 1 << v
            placed = False
            for i, (mask, typ) in enumerate(parts):
                nm = mask | bit
                if typ == 2:
                    if is_clique[nm]:
                        parts[i] = (nm, 1)
                        placed = True
                        break
                    elif is_indep[nm]:
                        parts[i] = (nm, 0)
                        placed = True
                        break
                elif typ == 1:
                    if is_clique[nm]:
                        parts[i] = (nm, 1)
                        placed = True
                        break
                else: # typ 0
                    if is_indep[nm]:
                        parts[i] = (nm, 0)
                        placed = True
                        break
            if not placed:
                parts.append((bit, 2))
        
        res = []
        for mask, _ in parts:
            part_list = [i for i in range(n) if (mask >> i) & 1]
            res.append(part_list)
        return res

    best_parts = greedy_partition()
    
    # DP
    INF = 10**9
    dp = [INF] * size
    dp[0] = 0
    parent = [-1] * size
    
    for S in range(1, size):
        if time.time() > deadline:
            break
            
        best_cost = INF
        best_T = -1
        T = S
        while T:
            if valid[T]:
                cost = 1 + dp[S ^ T]
                if cost < best_cost:
                    best_cost = cost
                    best_T = T
            T = (T - 1) & S
            
        dp[S] = best_cost
        parent[S] = best_T
        
    # Check if we finished DP for full set
    full_mask = (1 << n) - 1
    if time.time() <= deadline and dp[full_mask] < INF:
        # Reconstruct DP solution
        S = full_mask
        dp_parts = []
        while S:
            T = parent[S]
            part = []
            for i in range(n):
                if T & (1 << i):
                    part.append(i)
            dp_parts.append(part)
            S ^= T
        # DP solution is optimal, so better or equal to greedy
        # But we must ensure it's valid. DP guarantees it.
        return dp_parts
    else:
        # Return greedy solution
        return best_parts
