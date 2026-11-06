def construct():
    n = 16
    edges = []
    
    # 1. Base Graph: Grötzsch Graph (Mycielski of C5)
    # Vertices 0-4: C5 cycle
    # Vertices 5-9: Copies of 0-4
    # Vertex 10: Root
    # C5
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski step
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    for i in range(5):
        edges.append([5 + i, 10])
        
    # Initialize adjacency masks
    # adj_mask[i] is a bitmask where bit j is 1 if (i, j) is an edge
    adj_mask = [0] * n
    
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Function to find Maximum Independent Set (MIS) of the graph induced by vertices 0..limit-1
    # Using backtracking with bitmask
    def find_mis(limit):
        # We consider vertices 0 to limit-1
        # We want to find a subset S of {0..limit-1} that is independent and max size
        best_size = 0
        best_set = 0
        
        # Memoization or simple recursion with pruning
        # State: (current_vertex, current_set_mask)
        # We can just do a simple DFS
        
        def dfs(start, current_set, current_size):
            nonlocal best_size, best_set
            
            # Pruning: even if we add all remaining vertices, we can't beat best
            # Max possible size = current_size + (limit - start)
            if current_size + (limit - start) <= best_size:
                return
            
            if start == limit:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            # Option 1: Exclude vertex start
            dfs(start + 1, current_set, current_size)
            
            # Option 2: Include vertex start
            # Check if start is independent from current_set
            # current_set contains bits for vertices < start (since we process in order)
            # Wait, current_set can contain any vertex < start.
            # We need to check if adj_mask[start] & current_set == 0
            if (adj_mask[start] & current_set) == 0:
                dfs(start + 1, current_set | (1 << start), current_size + 1)
        
        dfs(0, 0, 0)
        return best_set

    # Add vertices 11 to 15
    for v in range(11, n):
        # Find MIS of the graph induced by 0..v-1
        mis_mask = find_mis(v)
        
        # Connect v to all vertices in mis_mask
        # Extract vertices from mask
        mask = mis_mask
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            u = lsb.bit_length() - 1
            edges.append([v, u])
            adj_mask[v] |= (1 << u)
            adj_mask[u] |= (1 << v)
            mask ^= lsb
            
    return edges
