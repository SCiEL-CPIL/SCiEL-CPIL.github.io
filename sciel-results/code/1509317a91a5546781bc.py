def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 26
    all_edges = list(itertools.combinations(range(n), 5))
    
    best_edges = None
    best_len = 51
    
    # Try to find a good instance
    for attempt in range(20):
        if time.time() > deadline - 20:
            break
            
        # Find a base instance
        base_edges = None
        # Try 50 edges first?
        for m in [50, 51]:
            for restart in range(200):
                if time.time() > deadline - 20:
                    break
                cand = random.sample(all_edges, m)
                if not is_2_colorable(n, cand):
                    base_edges = cand
                    break
            if base_edges:
                break
                
        if not base_edges:
            continue
            
        # Minimize
        current_edges = base_edges[:]
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if time.time() > deadline - 20:
                break
            e = current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            if not is_2_colorable(n, test_edges):
                current_edges = test_edges
                continue
            else:
                i += 1
                
        if len(current_edges) < best_len:
            best_len = len(current_edges)
            best_edges = current_edges[:]
            
    if best_edges:
        return {'n': n, 'edges': best_edges}
        
    # Fallback
    raise Exception("Failed")
