    def solve_hitting_set(n, cliques, time_limit_per_instance=30):
        if not cliques:
            return []
            
        # Precompute
        vertex_to_cliques = {v: [] for v in range(n)}
        clique_to_vertices = []
        
        for idx, c in enumerate(cliques):
            clique_to_vertices.append(list(c))
            for v in c:
                vertex_to_cliques[v].append(idx)
        
        num_cliques = len(cliques)
        
        def greedy_hitting_set(seed_offset=0):
            rng = random.Random(seed_offset)
            uncovered = set(range(num_cliques))
            selected = []
            
            while uncovered:
                counts = {v: 0 for v in range(n)}
                for c_idx in uncovered:
                    for v in clique_to_vertices[c_idx]:
                        counts[v] += 1
                
                max_count = max(counts.values())
                if max_count == 0:
                    break
                
                candidates = [v for v in range(n) if counts[v] == max_count]
                v = rng.choice(candidates)
                
                selected.append(v)
                
                new_uncovered = set()
                for c_idx in uncovered:
                    if v not in clique_to_vertices[c_idx]:
                        new_uncovered.add(c_idx)
                uncovered = new_uncovered
            
            # Local improvement with coverage counts
            # coverage_count[c] = number of selected vertices in clique c
            coverage_count = [0] * num_cliques
            for v in selected:
                for c_idx in vertex_to_cliques[v]:
                    coverage_count[c_idx] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected)):
                    v = selected[i]
                    # Check if v is redundant
                    # i.e., for all c in vertex_to_cliques[v], coverage_count[c] > 1
                    redundant = True
                    for c_idx in vertex_to_cliques[v]:
                        if coverage_count[c_idx] <= 1:
                            redundant = False
                            break
                            
                    if redundant:
                        # Remove v
                        for c_idx in vertex_to_cliques[v]:
                            coverage_count[c_idx] -= 1
                        selected.pop(i)
                        changed = True
                        break # Restart loop
            
            return selected

        best_solution = None
        best_size = float('inf')
        
        attempts = 0
        while time.time() < time_limit_per_instance:
            sol = greedy_hitting_set(attempts)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
            attempts += 1
            
            if attempts > 100 and best_size <= 3:
                break
                
        if best_solution is None:
            return []
        return sorted(best_solution)
