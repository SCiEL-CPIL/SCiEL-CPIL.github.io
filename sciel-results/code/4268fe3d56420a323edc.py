def construct():
    # Dense diameter-2 graph on 16 vertices where every edge is critical.
    # We start from K_{8,8} (64 edges, diameter 2, all edges critical) and
    # attempt to add intra-partition edges while maintaining:
    #   1. Diameter exactly 2
    #   2. Every edge is critical (removing it increases diameter above 2 or disconnects)
    
    import time
    deadline = time.time() + 80  # 80s safety margin
    
    n = 16
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    # Start with K_{8,8}
    base_edges = []
    for u in set_a:
        for v in set_b:
            base_edges.append([u, v])
    
    # Build adjacency for quick lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Candidate edges to try adding (intra-partition edges)
    candidates = []
    for i in range(8):
        for j in range(i+1, 8):
            candidates.append([set_a[i], set_a[j]])
            candidates.append([set_b[i], set_b[j]])
    
    # Try adding edges greedily, checking criticality
    current_edges = list(base_edges)
    
    def compute_diameter(edges):
        """Compute diameter using BFS from each vertex."""
        adj_local = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj_local[u][v] = True
            adj_local[v][u] = True
        
        # Check connectivity and compute distances
        max_dist = 0
        for src in range(n):
            # BFS
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in range(n):
                    if adj_local[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check if all vertices are reachable
            for d in dist:
                if d == -1:
                    return float('inf')
                max_dist = max(max_dist, d)
        return max_dist
    
    def is_edge_critical_fast(adj_mat, edge):
        """Check if removing 'edge' from the graph with adjacency 'adj_mat' 
        increases distance between its endpoints above 2 or disconnects them.
        
        An edge (u,v) is critical if and only if there is NO vertex w != u,v 
        such that (u,w) and (w,v) are both edges in the remaining graph.
        (Since we're checking if dist(u,v) > 2 after removing edge (u,v),
        and in a diameter-2 graph, the only way dist becomes >2 is if no 
        length-2 path exists.)
        """
        u, v = edge
        # Check if there's any vertex w that provides an alternative path of length 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[u][w] and adj_mat[w][v]:
                return False  # Alternative path exists, edge is NOT critical
        return True  # No alternative path, edge IS critical
    
    def check_all_edges_critical(adj_mat, edges):
        """Check all edges are critical using the fast method."""
        for e in edges:
            if not is_edge_critical_fast(adj_mat, e):
                return False
        return True
    
    # Try adding each candidate edge if it preserves the property
    for cand in candidates:
        if time.time() > deadline:
            break
        u, v = cand
        # Skip if edge already exists
        if adj[u][v]:
            continue
        
        # Try adding this edge
        test_edges = current_edges + [[u, v]]
        
        # Check diameter is still 2 (use fast check for diameter)
        # In a graph built from K_{8,8} plus intra-partition edges, 
        # diameter is 2 if all vertices are connected and no pair has dist > 2
        # Fast diameter check: verify connectivity and that no pair needs dist 3
        adj_test = [[False] * n for _ in range(n)]
        for eu, ev in test_edges:
            adj_test[eu][ev] = True
            adj_test[ev][eu] = True
        
        # Quick diameter check: for each vertex, BFS up to depth 2
        diam_ok = True
        for src in range(n):
            visited = [False] * n
            visited[src] = True
            # Depth 0
            frontier = [src]
            # Depth 1
            next_frontier = []
            for u in frontier:
                for v in range(n):
                    if adj_test[u][v] and not visited[v]:
                        visited[v] = True
                        next_frontier.append(v)
            if len(visited) - 1 == 0:
                # Only src is visited, need depth 2
                pass
            # Depth 2
            for u in next_frontier:
                for v in range(n):
                    if adj_test[u][v] and not visited[v]:
                        visited[v] = True
            if not all(visited):
                diam_ok = False
                break
        
        if not diam_ok:
            continue
        
        # Check all edges are critical in the new graph
        if not check_all_edges_critical(adj_test, test_edges):
            continue
        
        # Accept this edge
        current_edges = test_edges
        adj[u][v] = True
        adj[v][u] = True
    
    return current_edges
