import math
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = P['n']
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    LCM = 1
    for i in range(1, n + 1):
        LCM = lcm(LCM, i)
        
    best_m = 0
    best_L = -1
    indices = list(range(1, n + 1))
    
    # Hopcroft-Karp implementation
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] contains neighbors of left node i
        n_left: number of left nodes (n)
        n_right: number of right nodes (candidates)
        Returns: size of maximum matching
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
            
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching

    for m in range(LCM):
        # Calculate min_L
        first_multiples = [((m // i) + 1) * i for i in indices]
        min_L = max(x - m for x in first_multiples)
        
        # Max L is bounded by 144 according to problem statement, 
        # but we should verify if a match exists.
        # We iterate L from min_L to min_L + 144 (or just up to a safe limit)
        # The problem says "raw value ... is at most 144". 
        # So we only need to check L in [min_L, 144].
        
        found_match = False
        for L in range(min_L, 145):
            # Build graph for this m, L
            # Left nodes: indices 0..n-1 (representing 1..n)
            # Right nodes: candidates in (m, m+L]
            
            # Map candidate values to right-node indices
            # Candidates are values v such that m < v <= m+L and v % i == 0 for some i.
            # Actually, for the graph, we just need to know which right node corresponds to which value.
            # We can assign an ID to each unique candidate value.
            
            # Since L is small, we can just create a dict: value -> id
            # Or better, since candidates for different i overlap, we need a global ID for values.
            
            # Let's collect all unique candidates
            all_cands = set()
            for i in indices:
                start_val = first_multiples[i-1]
                if start_val <= m + L:
                    val = start_val
                    while val <= m + L:
                        all_cands.add(val)
                        val += i
            
            if not all_cands:
                continue
                
            # Sort candidates to have stable IDs (not strictly necessary for HK, but good)
            sorted_cands = sorted(all_cands)
            val_to_id = {v: idx for idx, v in enumerate(sorted_cands)}
            n_right = len(sorted_cands)
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for i_idx, i_val in enumerate(indices):
                start_val = first_multiples[i_idx]
                if start_val <= m + L:
                    val = start_val
                    while val <= m + L:
                        adj[i_idx].append(val_to_id[val])
                        val += i_val
            
            # Run Hopcroft-Karp
            match_size = hopcroft_karp(adj, n, n_right)
            
            if match_size == n:
                found_match = True
                if L > best_L:
                    best_L = L
                    best_m = m
                break
        
        # If no match found in 144, it's theoretically impossible per problem statement,
        # but if it happens, we just skip this m.
        
    return int(best_m)
