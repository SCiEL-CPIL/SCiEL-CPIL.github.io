    def has_independent_set_10(adj, n):
        # Greedy try
        # Pick vertex with min degree, add, remove neighbors
        # Repeat 10 times
        # If we get 10, return True
        # This is not exact, but a good filter.
        
        # Actually, for exactness, we must use B&B.
        # But we can order vertices in B&B by degree to prune better.
        
        # Let's stick to simple B&B for correctness.
        found_10 = False
        def dfs(candidates, current_size):
            nonlocal found_10
            if found_10:
                return
            if current_size + candidates.bit_count() < 10:
                return
            if current_size >= 10:
                found_10 = True
                return
            if candidates == 0:
                return
            # Heuristic: pick vertex with highest degree in induced subgraph
            # This helps prune the "exclude" branch?
            # Or pick lowest index?
            # Let's pick lowest index for simplicity.
            v = (candidates & -candidates).bit_length() - 1
            new_cands = candidates & ~adj[v] & ~(1 << v)
            dfs(new_cands, current_size + 1)
            if not found_10:
                new_cands = candidates & ~(1 << v)
                dfs(new_cands, current_size)
                
        dfs((1 << n) - 1, 0)
        return found_10
