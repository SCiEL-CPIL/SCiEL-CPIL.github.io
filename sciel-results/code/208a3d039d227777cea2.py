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
    
    def is_edge_critical(edges, edge):
        """Check if removing 'edge' from 'edges' increases diameter above 2 or disconnects."""
        u, v = edge
        # Remove edge
        remaining = [e for e in edges if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
        diam = compute_diameter(remaining)
        return diam > 2 or diam == float('inf')
    
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
        
        # Check diameter is still 2
        diam = compute_diameter(test_edges)
        if diam != 2:
            continue
        
        # Check all edges are critical in the new graph
        all_critical = True
        for e in test_edges:
            if not is_edge_critical(test_edges, e):
                all_critical = False
                break
        
        if all_critical:
            # Accept this edge
            current_edges = test_edges
            adj[u][v] = True
            adj[v][u] = True
    
    return current_edges
