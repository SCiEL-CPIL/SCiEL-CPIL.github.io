    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build incompatibility graph and neighbors
        incompatible = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        neighbors = [list(incompatible[i]) for i in range(m)]
        
        # Initial Greedy Coloring
        degrees = [len(neighbors[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: -degrees[x])
        
        colors_greedy = [-1] * m
        current_color_greedy = 0
        
        for v in order:
            used_colors = set()
            for u in neighbors[v]:
                if colors_greedy[u] != -1:
                    used_colors.add(colors_greedy[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors_greedy[v] = c
            if c + 1 > current_color_greedy:
                current_color_greedy = c + 1
        
        best_colors = colors_greedy[:]
        best_count = current_color_greedy
        
        # DSATUR Backtracking Search
        def search_with_limit(limit):
            colors = [-1] * m
            n_colors = [set() for _ in range(m)]
            
            def backtrack():
                nonlocal deadline
                if time.time() > deadline:
                    return None
                
                # Find uncolored vertex with max saturation
                best_v = -1
                best_sat = -1
                best_deg = -1
                
                for v in range(m):
                    if colors[v] == -1:
                        s = len(n_colors[v])
                        d = len(neighbors[v])
                        if s > best_sat or (s == best_sat and d > best_deg):
                            best_sat = s
                            best_deg = d
                            best_v = v
                            
                if best_v == -1:
                    return colors[:] # Success
                
                v = best_v
                
                # Try colors
                available_colors = [c for c in range(limit) if c not in n_colors[v]]
                
                if not available_colors:
                    return None
                
                for c in available_colors:
                    colors[v] = c
                    for u in neighbors[v]:
                        n_colors[u].add(c)
                    
                    result = backtrack()
                    if result is not None:
                        return result
                        
                    colors[v] = -1
                    for u in neighbors[v]:
                        n_colors[u].discard(c)
                        
                return None
                
            return backtrack()
            
        # Try to reduce the number of colors
        for limit in range(best_count - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
                
            result = search_with_limit(limit)
            if result is not None:
                best_colors = result
                best_count = limit
                
        return best_colors
