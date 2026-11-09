<<<<<<< SEARCH
def _local_search_polish(adj, time_limit):
    """
    Fast local search: try removing one edge and adding one or two edges.
    Used as a final polish after Simulated Annealing.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Precompute missing edges once per iteration
        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        
        # Try 1-remove, 2-add swap (net +1)
        for u, v in all_pairs:
            if time.time() - start_time > time_limit:
                return best_adj
            
            if not ((adj[u] >> v) & 1):
                continue
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            added = 0
            new_edges = []
            
            # Try to add up to 2 missing edges (in order of potential benefit)
            for u2, v2 in missing:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))
            
            if added >= 2:
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                # Update missing list incrementally
                for eu, ev in new_edges:
                    for i, (mu, mv) in enumerate(missing):
                        if (mu, mv) == (eu, ev) or (mu, mv) == (ev, eu):
                            missing.pop(i)
                            break
            else:
                # Revert
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if not improved:
            break
            
    return best_adj
=======
def _local_search_polish(adj, time_limit):
    """
    Fast local search: try removing one edge and adding one or two edges.
    Also tries 2-remove, 3-add for larger jumps.
    Used as a final polish after Simulated Annealing.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Precompute missing edges once per iteration
        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        
        # Try 1-remove, 2-add swap (net +1)
        for u, v in all_pairs:
            if time.time() - start_time > time_limit:
                return best_adj
            
            if not ((adj[u] >> v) & 1):
                continue
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            added = 0
            new_edges = []
            
            # Try to add up to 2 missing edges
            for u2, v2 in missing:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))
            
            if added >= 2:
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                # Rebuild missing list (simpler and correct)
                missing = [(u2, v2) for u2, v2 in all_pairs if not ((adj[u2] >> v2) & 1)]
            else:
                # Revert
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Try 2-remove, 3-add swap (net +1) - more aggressive
        if time.time() - start_time > time_limit:
            return best_adj
        
        existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        if len(existing) >= 2:
            # Sample a few 2-edge removals
            n_attempts = min(50, len(existing) * (len(existing) - 1) // 2)
            for _ in range(n_attempts):
                if time.time() - start_time > time_limit:
                    return best_adj
                
                # Pick two random existing edges
                i1 = random.randint(0, len(existing) - 1)
                i2 = random.randint(0, len(existing) - 2)
                if i2 >= i1:
                    i2 += 1
                
                e1 = existing[i1]
                e2 = existing[i2]
                
                # Remove both edges
                adj[e1[0]] &= ~(1 << e1[1])
                adj[e1[1]] &= ~(1 << e1[0])
                adj[e2[0]] &= ~(1 << e2[1])
                adj[e2[1]] &= ~(1 << e2[0])
                
                added = 0
                new_edges = []
                
                # Try to add up to 3 missing edges
                missing = [(u2, v2) for u2, v2 in all_pairs if not ((adj[u2] >> v2) & 1)]
                for u2, v2 in missing:
                    if added >= 3:
                        break
                    if _can_add_edge_fast(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        added += 1
                        new_edges.append((u2, v2))
                
                if added >= 3:
                    current_count += (added - 2)
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Revert
                    for eu, ev in new_edges:
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                    adj[e1[0]] |= (1 << e1[1])
                    adj[e1[1]] |= (1 << e1[0])
                    adj[e2[0]] |= (1 << e2[1])
                    adj[e2[1]] |= (1 << e2[0])
        
        if not improved:
            break
            
    return best_adj
>>>>>>> REPLACE

