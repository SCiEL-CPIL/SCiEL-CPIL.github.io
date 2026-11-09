<<<<<<< SEARCH
    # Phase 2: IDDFS from promising starts
    # We prioritize starts that are high degree or were good in greedy
    # Sort promising starts by degree descending to find "central" paths first
    unique_promising = list(set(promising_starts))
    # Add top degree nodes from the largest component if not already present
    deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
    for v in deg_sorted[:10]:
        if v not in unique_promising:
            unique_promising.append(v)
            
    # Limit the number of IDDFS runs to ensure we finish in time
    # Each IDDFS run can be expensive. 
    # We have ~50s left. Let's allow ~10-20 runs.
    max_iddfs_runs = 15
    
    for start in unique_promising[:max_iddfs_runs]:
        if time.time() > deadline - 10:
            break
            
        # Start IDDFS from a depth close to current best to save time
        # If we have a path of length L, we try to find L+1, L+2...
        # But IDDFS is more effective if we start lower. 
        # However, if we know a path of length 80 exists, starting IDDFS at 1 is wasteful.
        # Let's start at max(5, best_len - 5) to catch nearby improvements.
        # Actually, if we start at best_len, and we find a path of that length, we go to best_len+1.
        # If we don't, we stop. This is efficient.
        
        start_depth = max(5, best_len - 2)
        
        p = iddfs_search(start, ADJ, deadline - 10, initial_depth=start_depth)
        p = refine(p, ADJ)
        
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            
            if best_len == N:
                return best_path
=======
    # Phase 2: IDDFS from promising starts
    # We prioritize starts that are high degree or were good in greedy
    # Sort promising starts by degree descending to find "central" paths first
    unique_promising = list(set(promising_starts))
    # Add top degree nodes from the largest component if not already present
    deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
    for v in deg_sorted[:10]:
        if v not in unique_promising:
            unique_promising.append(v)
            
    # Limit the number of IDDFS runs to ensure we finish in time
    # Each IDDFS run can be expensive. 
    # We have ~50s left. Let's allow ~10-20 runs.
    max_iddfs_runs = 15
    
    for start in unique_promising[:max_iddfs_runs]:
        if time.time() > deadline - 10:
            break
            
        # Start IDDFS from a moderate depth to balance exploration and exploitation
        # Starting too high risks missing longer paths that need different initial segments
        # Starting too low wastes time on short paths we already know exist
        start_depth = max(5, best_len - 5)
        
        p = iddfs_search(start, ADJ, deadline - 10, initial_depth=start_depth)
        p = refine(p, ADJ)
        
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            
            if best_len == N:
                return best_path
>>>>>>> REPLACE
