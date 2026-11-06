def decompose(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    pieces = []
    remaining_edges = set(edges)
    
    # Sort vertices by degree to prioritize high-degree vertices in cycle finding
    # This often leads to longer cycles
    
    while remaining_edges:
        # Find a cycle
        best_cycle = None
        best_len = 0
        
        # Try to find a long cycle
        # We can limit the search to avoid O(V^2) if it's too slow
        # But n is small (30), so it's fine.
        
        for start in list(adj.keys()):
            if not adj[start]:
                continue
            
            # DFS to find a cycle
            visited = set()
            parent = {}
            path = []
            
            def dfs(u):
                nonlocal best_cycle, best_len
                visited.add(u)
                path.append(u)
                for v in adj[u]:
                    if v == parent.get(u):
                        continue
                    if v in visited:
                        if v in path:
                            idx = path.index(v)
                            cycle = path[idx:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                    else:
                        parent[v] = u
                        dfs(v)
                path.pop()
            
            dfs(start)
            
            # Optional: Early exit if we found a Hamiltonian cycle or very long one
            if best_len >= len(remaining_edges):
                break
        
        if best_cycle:
            # Remove edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
            pieces.append(best_cycle)
        else:
            # No cycle, take an edge
            edge = next(iter(remaining_edges))
            remaining_edges.discard(edge)
            u, v = edge
            adj[u].remove(v)
            adj[v].remove(u)
            pieces.append(list(edge))
            
    return pieces
