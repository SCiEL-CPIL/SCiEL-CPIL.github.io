    all_possible_edges = list(combinations(range(n), 2))
    base_edge_set = set(groetzsch_edges)
    
    for restart in range(200): # Up to 200 restarts
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(n)]
        for u, v in groetzsch_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        current_edges = list(groetzsch_edges)
        
        # Candidate edges: all edges not in base
        candidates = [e for e in all_possible_edges if e not in base_edge_set]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            if not (adj[u] & adj[v]):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
    