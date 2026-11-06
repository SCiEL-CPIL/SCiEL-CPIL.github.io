def construct():
    from random import Random
    import heapq

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Helper to get remaining neighbors of a vertex
        def get_remaining_neighbors(v):
            return [u for u in adj.get(v, []) if tuple(sorted((v, u))) in remaining_edges]
        
        # Helper to get remaining degree of a vertex
        def get_remaining_degree(v):
            return len(get_remaining_neighbors(v))

        while remaining_edges:
            # Find vertices with remaining edges
            active_vertices = [v for v in adj if get_remaining_degree(v) > 0]
            
            if not active_vertices:
                break

            # Heuristic: Start from a vertex with minimum degree (likely an endpoint)
            # If multiple, pick one that leads to a longer path? 
            # For now, just pick min degree. Tie-break by index for determinism.
            start_candidates = sorted(active_vertices, key=lambda v: (get_remaining_degree(v), v))
            
            best_path = None
            best_path_len = 0
            
            # Try a few start candidates (e.g., the 3 with lowest degree)
            # In small graphs, trying all is fine, but let's be safe and efficient.
            trial_starts = start_candidates[:3]
            
            for start in trial_starts:
                # Simulate a greedy walk
                path = [start]
                used_vertices = {start}
                current = start
                local_remaining = set(remaining_edges) # Copy for simulation
                
                while True:
                    neighbors = [u for u in adj.get(current, []) if tuple(sorted((current, u))) in local_remaining and u not in used_vertices]
                    if not neighbors:
                        break
                    
                    # Warnsdorff's rule: pick neighbor with smallest remaining degree
                    # Note: We must calculate degree in the context of local_remaining and used_vertices
                    def sim_degree(v):
                        return len([x for x in adj.get(v, []) if tuple(sorted((v, x))) in local_remaining and x not in used_vertices])
                    
                    neighbors.sort(key=sim_degree)
                    next_v = neighbors[0]
                    path.append(next_v)
                    used_vertices.add(next_v)
                    local_remaining.remove(tuple(sorted((current, next_v))))
                    current = next_v
                
                if len(path) > best_path_len:
                    best_path_len = len(path)
                    best_path = path
            
            if best_path:
                paths.append(best_path)
                # Commit: remove edges from global remaining_edges
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    remaining_edges.remove(tuple(sorted((u, v))))
            else:
                # Fallback: should not happen if remaining_edges is not empty
                u, v = next(iter(remaining_edges))
                paths.append([u, v])
                remaining_edges.remove((u, v))

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
