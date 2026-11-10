import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    size = 1 << n
    
    # Precompute is_clique and is_indep
    # is_clique[S] is True if S is a clique
    # is_indep[S] is True if S is an independent set
    is_clique = [False] * size
    is_indep = [False] * size
    valid = [False] * size
    
    # Empty set is both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    valid[0] = True
    
    for S in range(1, size):
        # Get least significant bit
        lsb = S & -S
        v = lsb.bit_length() - 1
        S_prime = S ^ lsb
        
        # Check clique: S is clique if S' is clique and v is connected to all in S'
        if is_clique[S_prime]:
            # Check if v is adjacent to all nodes in S_prime
            # adj[v] has 1s for neighbors. We need (adj[v] & S_prime) == S_prime
            if (adj[v] & S_prime) == S_prime:
                is_clique[S] = True
            else:
                is_clique[S] = False
        else:
            is_clique[S] = False
            
        # Check independent: S is indep if S' is indep and v is NOT connected to any in S'
        if is_indep[S_prime]:
            # Check if v has no neighbors in S_prime
            if (adj[v] & S_prime) == 0:
                is_indep[S] = True
            else:
                is_indep[S] = False
        else:
            is_indep[S] = False
            
        valid[S] = is_clique[S] or is_indep[S]
        
    # DP to find min number of parts
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    
    # To reconstruct, we store the best submask T for each S
    parent = [-1] * size
    
    for S in range(1, size):
        best_cost = INF
        best_T = -1
        # Iterate submasks of S
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
        
    # Reconstruct partition
    S = (1 << n) - 1
    parts = []
    while S:
        T = parent[S]
        # Convert bitmask T to list of vertices
        part = []
        for i in range(n):
            if T & (1 << i):
                part.append(i)
        parts.append(part)
        S ^= T
        
    return parts
