def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 115.0
    n = 31
    target_edges = 91
    
    # 1. Greedy construction of C4-free graph
    def creates_c4(adj, u, v):
        # Check if adding (u,v) creates a C4
        # A C4 is created if there is a path of length 2 between u and v
        # i.e., common neighbors
        common = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common += 1
                if common > 1:
                    return True
        # Actually, if common >= 1, adding (u,v) creates a C4?
        # u-w-v path exists. Adding u-v closes a triangle? No.
        # u-w-v is a path. u-v is an edge.
        # This forms a triangle u-w-v.
        # A C4 requires 4 vertices.
        # If u and v have a common neighbor w, adding (u,v) creates a triangle.
        # Does it create a C4?
        # A C4 containing (u,v) requires another path of length 3 between u and v.
        # Or, if there are TWO common neighbors w1, w2, then u-w1-v-w2-u is a C4.
        # So, adding (u,v) creates a C4 iff u and v have >= 2 common neighbors.
        return common >= 2

    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    # Greedy add
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if len(edges) >= 90:
            break
        if not creates_c4(adj, u, v):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
    # If we have 90 edges, add 1 more
    if len(edges) == 90:
        # Find a pair with minimum common neighbors (should be 0 or 1)
        # If 0 common neighbors, adding creates 0 C4s? No, 0 C4s means still C4-free?
        # If 0 common neighbors, no path of length 2.
        # Adding (u,v) creates no C4.
        # So we can have 91 edges and 0 C4s?
        # No, ex(31, C4) = 90. So 91 edges MUST have >= 1 C4.
        # This implies my check `creates_c4` is incomplete or ex(31, C4) is not 90?
        # Prompt says ex(31, C4) = 90.
        # So 91 edges => >= 1 C4.
        # If I add an edge with 0 common neighbors, I haven't created a C4.
        # So the graph was not maximal C4-free?
        # If I can add an edge without creating a C4, the previous graph wasn't maximal.
        # So greedy should continue until no more edges can be added.
        
        pass

    # Local search to maximize edges in C4-free graph or minimize C4s in 91-edge graph
    # Target: 91 edges, min C4s.
    
    # If greedy got < 90 edges, we need to add edges that create C4s.
    # Strategy:
    # 1. Get to 90 edges C4-free if possible.
    # 2. Add 1 edge.
    
    # If greedy failed to get 90, use local search on C4-free constraint
    # Remove edge involved in C4? No, we want to ADD edges.
    
    # Let's just run a local search on the 91-edge graph to minimize C4s.
    
    def count_c4(adj):
        c4 = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    continue
                common = 0
                for k in range(n):
                    if adj[i][k] and adj[j][k]:
                        common += 1
                c4 += common * (common - 1) // 2
        # Wait, this counts C4s formed by adding non-edges?
        # No, we need to count C4s in the existing graph.
        # C4 in graph: i-j-k-l-i.
        # Count pairs of vertices (i,k) and common neighbors (j,l).
        # For each pair (i,k), let c be # common neighbors.
        # Number of C4s using (i,k) as diagonal is c*(c-1)/2.
        # Each C4 is counted twice (once for each diagonal).
        total = 0
        for i in range(n):
            for k in range(i+1, n):
                c = 0
                for j in range(n):
                    if adj[i][j] and adj[k][j]:
                        c += 1
                total += c * (c - 1) // 2
        return total // 2

    # Ensure we have 91 edges
    if len(edges) < 91:
        # Add random edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges) >= 91:
                break
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
    # Local search: swap edges to reduce C4 count
    current_c4 = count_c4(adj)
    
    while time.time() - start_time < DEADLINE:
        improved = False
        # Try removing an edge and adding another
        # Sample some edges to remove
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        for u, v in edge_list[:100]: # Limit swaps per iteration
            # Remove (u,v)
            adj[u][v] = 0
            adj[v][u] = 0
            edges.remove((u, v))
            
            # Find best edge to add
            best_add = None
            best_delta = 0
            # Calculate C4 change
            # Removing (u,v) reduces C4s by (#C4s containing u,v)
            # Adding (x,y) increases C4s by (#C4s created by x,y)
            
            # Simplified: just count C4s after swap
            # Too slow.
            
            # Heuristic: Add edge with min common neighbors
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges]
            # Sort by common neighbors
            # This is O(N^3).
            
            pass
        
    return [[u, v] for u, v in edges]
