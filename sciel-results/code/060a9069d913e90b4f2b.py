<<<<<<< SEARCH
def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g.
    Uses a girth-aware BFS to pick stub pairs that avoid short cycles."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    # Try a range of n values, starting from lower bound
    max_target = min(lower + 200, 600)  # cap to keep search tractable
    for offset in range(0, 300):
        if time.time() - start_time > time_budget:
            break
        target_n = lower + offset
        if target_n < k + 1:
            continue
        if k * target_n % 2 != 0:
            continue
        if target_n > max_target:
            break
        
        # Try multiple random initializations with girth-aware configuration
        attempts = 8 if target_n <= lower + 50 else 4
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            
            adj = _random_k_regular_girth_aware(target_n, k, g)
            if adj is None:
                continue
            
            # Verify girth
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break  # Found valid for this target_n
    
    if best_n is not None:
        return best_n, best_edges
    return None, None
=======
def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    max_target = min(lower + 300, 800)
    
    for offset in range(0, 400):
        if time.time() - start_time > time_budget:
            break
        target_n = lower + offset
        if target_n < k + 1:
            continue
        if k * target_n % 2 != 0:
            continue
        if target_n > max_target:
            break
        
        # More attempts for smaller targets
        attempts = 12 if target_n <= lower + 30 else (8 if target_n <= lower + 100 else 4)
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            
            adj = _random_k_regular_girth_aware(target_n, k, g, max_attempts=15)
            if adj is None:
                continue
            
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break
    
    if best_n is not None:
        return best_n, best_edges
    return None, None
>>>>>>> REPLACE

