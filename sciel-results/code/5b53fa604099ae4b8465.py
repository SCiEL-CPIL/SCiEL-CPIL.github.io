<<<<<<< SEARCH
    # Initial best
    best_raw = 3
    best_edges = None
    
    # Try to find a graph with raw value 1
    # We will use a simple random restart local search.
    
    max_iterations = 500000
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        # Generate a random starting point
        edges = random_graph()
        
        if not is_valid(edges):
            continue
            
        lengths = get_cycle_lengths(edges)
        if not lengths:
            continue
            
        raw = len(lengths)
        
        if raw < best_raw:
            best_raw = raw
            best_edges = [e[:] for e in edges]
            if best_raw == 1:
                break
            
        # Local search: try to improve
        # Since we only care about raw value 1, if we are at 2, we try to get to 1.
        # If we are at 3, we try to get to 2 or 1.
        
        # We'll do a few steps of local search
        for step in range(20):
            if time.time() - start_time > timeout:
                break
                
            new_edges = mutate(edges, build_adj(edges))
            if not is_valid(new_edges):
                continue
                
            new_lengths = get_cycle_lengths(new_edges)
            if not new_lengths:
                continue
                
            new_raw = len(new_lengths)
            
            if new_raw < raw:
                raw = new_raw
                edges = new_edges
                if raw < best_raw:
                    best_raw = raw
                    best_edges = [e[:] for e in edges]
                    if best_raw == 1:
                        break
            elif new_raw == raw and random.random() < 0.5:
                # Accept sideways moves to escape local minima
                edges = new_edges
                
        if best_raw == 1:
            break
=======
    # Initial best
    best_raw = 3
    best_edges = None
    
    # Try to find a graph with raw value 1
    # We will use a simple random restart local search.
    
    all_vertices = list(range(N))
    
    max_iterations = 200000
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        # Generate a random starting point
        edges = random_graph()
        
        if not is_valid(edges):
            continue
            
        lengths = get_cycle_lengths(edges)
        if not lengths:
            continue
            
        raw = len(lengths)
        
        if raw < best_raw:
            best_raw = raw
            best_edges = [e[:] for e in edges]
            if best_raw == 1:
                break
            
        # Local search: try to improve
        # Since we only care about raw value 1, if we are at 2, we try to get to 1.
        # If we are at 3, we try to get to 2 or 1.
        
        # We'll do a few steps of local search
        adj_set = build_adj(edges)
        for step in range(30):
            if time.time() - start_time > timeout:
                break
                
            new_edges = mutate(edges, adj_set)
            if not is_valid(new_edges):
                continue
                
            new_lengths = get_cycle_lengths(new_edges)
            if not new_lengths:
                continue
                
            new_raw = len(new_lengths)
            
            if new_raw < raw:
                raw = new_raw
                edges = new_edges
                adj_set = build_adj(edges)
                if raw < best_raw:
                    best_raw = raw
                    best_edges = [e[:] for e in edges]
                    if best_raw == 1:
                        break
            elif new_raw == raw and random.random() < 0.3:
                # Accept sideways moves to escape local minima
                edges = new_edges
                adj_set = build_adj(edges)
                
        if best_raw == 1:
            break
>>>>>>> REPLACE

