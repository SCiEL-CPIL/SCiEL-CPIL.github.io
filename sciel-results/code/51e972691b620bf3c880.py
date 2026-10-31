<<<<<<< SEARCH
    # Strategy: Try Petersen lifts with increasing m, checking girth.
    # Expected girth of random m-lift of Petersen grows like c*log(m).
    # For girth 16, we need m large enough. Let's try m from 200 upward.
    
    rng = random.Random(42)
    best_result = None

    # Phase 1: Try Petersen lifts with various m values
    m_values = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        n = 10 * m
        if best_result is not None and n >= best_result[0]:
            continue
        
        num_tries = max(2, 10 - m // 500)
        for attempt in range(num_tries):
            if time.time() > deadline - 10:
                break
            
            adj = build_petersen_lift(m, rng)
            if adj is None:
                continue
            
            # Quick sampled check first
            if check_girth_sampled(adj, n, g, min(n, 200), rng):
                # Full check
                if check_girth_full(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_result = (n, edges)
                    break  # found one for this m, try smaller m next
    
    # Phase 2: If no Petersen lift worked, try random 3-regular graphs
    if best_result is None and time.time() < deadline - 10:
        for n_try in [400, 500, 600, 800, 1000, 1200, 1500]:
            if time.time() > deadline - 10:
                break
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                adj = build_random_3reg(n_try, rng)
                if adj is None:
                    continue
                if check_girth_sampled(adj, n_try, g, min(n_try, 100), rng):
                    if check_girth_full(adj, n_try, g):
                        edges = []
                        for u in range(n_try):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_result = (n_try, edges)
                        break
            if best_result is not None:
                break
=======
    # Strategy: Try K_{3,3} lifts (n=6m) and Petersen lifts (n=10m) with increasing m.
    # K_{3,3} lifts give smaller n for the same m, so try them first.
    
    rng = random.Random(42)
    best_result = None

    # Phase 1: Try K_{3,3} lifts with various m values (n = 6m)
    m_values_k33 = [100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]
    
    for m in m_values_k33:
        if time.time() > deadline - 10:
            break
        n = 6 * m
        if best_result is not None and n >= best_result[0]:
            continue
        
        num_tries = max(3, 15 - m // 100)
        for attempt in range(num_tries):
            if time.time() > deadline - 10:
                break
            
            adj = build_k33_lift(m, rng)
            if adj is None:
                continue
            
            # Quick sampled check first
            if check_girth_sampled(adj, n, g, min(n, 200), rng):
                # Full check
                if check_girth_full(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_result = (n, edges)
                    break
    
    # Phase 2: Try Petersen lifts with various m values (n = 10m)
    m_values_petersen = [100, 150, 200, 300, 400, 500, 750, 1000]
    
    for m in m_values_petersen:
        if time.time() > deadline - 10:
            break
        n = 10 * m
        if best_result is not None and n >= best_result[0]:
            continue
        
        num_tries = max(3, 12 - m // 100)
        for attempt in range(num_tries):
            if time.time() > deadline - 10:
                break
            
            adj = build_petersen_lift2(m, rng)
            if adj is None:
                continue
            
            if check_girth_sampled(adj, n, g, min(n, 200), rng):
                if check_girth_full(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_result = (n, edges)
                    break
    
    # Phase 3: If no lift worked, try random 3-regular graphs
    if best_result is None and time.time() < deadline - 10:
        for n_try in [400, 500, 600, 800, 1000, 1200, 1500]:
            if time.time() > deadline - 10:
                break
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                adj = build_random_3reg(n_try, rng)
                if adj is None:
                    continue
                if check_girth_sampled(adj, n_try, g, min(n_try, 100), rng):
                    if check_girth_full(adj, n_try, g):
                        edges = []
                        for u in range(n_try):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_result = (n_try, edges)
                        break
            if best_result is not None:
                break
>>>>>>> REPLACE

