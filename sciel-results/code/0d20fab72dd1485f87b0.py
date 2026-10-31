def _find_mis_supports(adj, time_limit=10.0):
    """Find Maximum Independent Set using Greedy + Local Search."""
    n = adj.shape[0]
    start_time = time.time()
    
    best_size = 0
    best_set = set()
    
    rng = np.random.default_rng(42)
    
    # Precompute neighbor lists for speed
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Phase: Random order, pick if no conflict with already picked
        alive = np.ones(n, dtype=bool)
        selected = set()
        order = rng.permutation(n)
        
        for idx in order:
            if not alive[idx]:
                continue
            # Check if idx conflicts with any in selected
            # Since we remove neighbors from 'alive', if alive[idx] is True, 
            # it means it wasn't removed by a previous selection? 
            # Wait, the previous code removed neighbors from 'alive'.
            # So if alive[idx] is True, it is safe to pick.
            selected.add(idx)
            # Remove idx and its neighbors from alive
            to_remove = np.union1d([idx], neighbors[idx])
            alive[to_remove] = False
            
        current_set = selected
        current_size = len(current_set)
        
        # Local Search Phase: Try 1-2 swaps
        # Try to remove one node and add two that become available
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Calculate which nodes are "blocked" by current_set
            # A node u is blocked if it has a neighbor in current_set
            blocked_by = [set() for _ in range(n)]
            for v in current_set:
                for u in neighbors[v]:
                    if u not in current_set:
                        blocked_by[u].add(v)
            
            # Try each swap: remove v, add u1, u2
            # Optimization: Only try swaps where u1, u2 are blocked ONLY by v
            # This is complex. Simpler: Iterate through all v in current_set.
            # For each v, find candidates u that are blocked by v.
            # If removing v allows u1 and u2 to be added, and they don't conflict, do it.
            
            for v in list(current_set):
                # Find nodes blocked by v
                candidates = [u for u in neighbors[v] if u not in current_set and blocked_by[u] == {v}]
                
                if len(candidates) < 2:
                    continue
                
                # Try all pairs of candidates
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        u1 = candidates[i]
                        u2 = candidates[j]
                        
                        # Check if u1 and u2 conflict
                        if adj[u1, u2]:
                            continue
                        
                        # Check if u1 or u2 conflict with other nodes in current_set \ {v}
                        # They are blocked by v, so if we remove v, are they free?
                        # Yes, if blocked_by[u] == {v}.
                        # But we must ensure they don't conflict with each other (checked)
                        # and that removing v doesn't break anything else (it doesn't, we are adding).
                        
                        # Perform swap
                        new_set = current_set - {v} | {u1, u2}
                        
                        # Verify independence (safety check, should be true by logic)
                        is_independent = True
                        for x in new_set:
                            for y in new_set:
                                if x != y and adj[x, y]:
                                    is_independent = False
                                    break
                            if not is_independent:
                                break
                        
                        if is_independent and len(new_set) > current_size:
                            current_set = new_set
                            current_size = len(current_set)
                            improved = True
                            # Restart local search with new set
                            break
                    if improved:
                        break
                if improved:
                    break
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
    return list(best_set)
