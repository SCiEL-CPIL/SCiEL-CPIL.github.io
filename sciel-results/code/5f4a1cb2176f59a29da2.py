def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 40
    best_edges_count = 0
    best_edges = []
    
    # Initial K5s
    base_edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                base_edges.append([5 * k + a, 5 * k + b])
                
    potential_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if u // 5 != v // 5:
                potential_edges.append((u, v))
                
    # Shuffle potential edges once? Or per restart?
    # Per restart is better for diversity.
    
    while time.time() < deadline - 5: # 5s margin for final processing
        adj = [set() for _ in range(n)]
        edges = base_edges[:]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Shuffle
        shuffled = potential_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time.time() > deadline - 5:
                break
            
            # Check C6
            if has_simple_path_len4(u, v, adj):
                continue
                
            # Add edge
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
            
        if len(edges) > best_edges_count:
            best_edges_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
