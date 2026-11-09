import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate Grötzsch Graph (Mycielski of C5)
    # Vertices: 0..10
    # Edges: 20
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 0], [5, 2],                          # Copy 5 of 0
        [6, 1], [6, 3],                          # Copy 6 of 1
        [7, 2], [7, 4],                          # Copy 7 of 2
        [8, 3], [8, 0],                          # Copy 8 of 3
        [9, 4], [9, 1],                          # Copy 9 of 4
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9] # Top 10
    ]
    
    adj = [set() for _ in range(16)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_max_independent_set(adj, n):
        """
        Find Maximum Independent Set in graph induced by vertices 0..n-1.
        Uses backtracking with pruning.
        """
        best_set = []
        best_size = 0
        
        # Pre-compute degrees for ordering heuristic
        # We will sort 'available' dynamically in the recursion if needed, 
        # but for n<=16, simple ordering is fine.
        
        def backtrack(current_set, available):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
                
            if not available:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Upper bound: current size + remaining available
            if len(current_set) + len(available) <= best_size:
                return
                
            # Heuristic: Pick vertex with highest degree in the subgraph?
            # Or just pick the first one.
            # To improve pruning, we can sort available by degree descending.
            # However, modifying 'available' list in place or creating new lists 
            # has overhead. For n=16, it's negligible.
            
            v = available[0]
            remaining = available[1:]
            
            # Option 1: Include v
            current_set.append(v)
            # New available: vertices in remaining that are not neighbors of v
            new_available = [u for u in remaining if u not in adj[v]]
            backtrack(current_set, new_available)
            current_set.pop()
            
            # Option 2: Exclude v
            backtrack(current_set, remaining)
            
        # Initial available set: 0 to n-1
        # Order by degree (descending) might help pruning
        initial_available = list(range(n))
        initial_available.sort(key=lambda x: len(adj[x]), reverse=True)
        
        backtrack([], initial_available)
        return best_set

    # 2. Add vertices 11 to 15 sequentially
    # Connect each new vertex to the Maximum Independent Set of the current graph
    for v in range(11, 16):
        if time.time() > deadline:
            break
            
        current_n = v
        # Find MIS in vertices 0..current_n-1
        mis = get_max_independent_set(adj, current_n)
        
        # Connect v to all vertices in MIS
        for u in mis:
            edges.append([v, u])
            adj[v].add(u)
            adj[u].add(v)
            
    # 3. Verify triangle-free
    is_valid = True
    for i in range(16):
        for j in adj[i]:
            if i > j:
                continue
            common = adj[i] & adj[j]
            if common:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback: return the core Grötzsch graph (should not happen)
        return edges[:20]

    return edges
