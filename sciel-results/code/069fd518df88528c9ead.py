<<<<<<< SEARCH
        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        graph_time = min(remaining_time * 0.9, 40.0)
        
        pieces = _decompose_greedy(edges, n, graph_time)
=======
        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        graph_time = min(remaining_time * 0.9, 40.0)
        
        # Find all short cycles
        max_cycle_len = 15
        cycles = _find_all_short_cycles(adj, n, max_cycle_len, edges_set, start_time, graph_time)
        
        # Sort cycles by length descending (higher savings)
        cycles.sort(key=lambda c: len(c), reverse=True)
        
        selected_cycles = []
        used_edges = set()
        
        for cycle in cycles:
            # Check if edges are available
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
            if valid:
                selected_cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    used_edges.add(e)
        
        # Build pieces from selected cycles
        pieces = [c for c in selected_cycles]
        
        # Add remaining edges
        for e in edges_set:
            if e not in used_edges:
                pieces.append(list(e))
>>>>>>> REPLACE
