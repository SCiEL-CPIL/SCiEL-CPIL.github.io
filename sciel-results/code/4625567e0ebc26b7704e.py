<<<<<<< SEARCH
    # Start with all clockwise
    best_routing = np.zeros(N_DEMANDS)
    best_max = _max_load(best_routing)
    
    rng = np.random.default_rng(42)
    
    # Method 1: 2-bit local search from all-zero
    routing, max_load = _local_search_2bit(np.zeros(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 2: 2-bit local search from all-one
    routing, max_load = _local_search_2bit(np.ones(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 3: Many random restarts with 2-bit local search
    for trial in range(200):
        if time.time() > deadline - 120:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        routing, max_load = _local_search_2bit(random_routing, max_iters=20000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: Simulated annealing from best found so far
    if time.time() < deadline - 120:
        routing, max_load = _simulated_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=30000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4b: Simulated annealing with soft-max surrogate
    if time.time() < deadline - 100:
        routing, max_load = _simulated_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=30000, use_soft=True, soft_beta=50.0)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: Tabu search from best found so far
    if time.time() < deadline - 90:
        routing, max_load = _tabu_search(best_routing, tabu_len=20, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 6: More random restarts with SA + tabu
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        # First 2-bit local search
        hc_routing, hc_max = _local_search_2bit(random_routing, max_iters=10000)
        # Then SA from there
        sa_routing, sa_max = _simulated_annealing(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=15000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
        
        # Then tabu from there
        if time.time() < deadline - 30:
            ts_routing, ts_max = _tabu_search(sa_routing, tabu_len=15, max_iters=20000)
            if ts_max < best_max:
                best_max = ts_max
                best_routing = ts_routing.copy()
    
    # Method 7: 3-bit local search from best
    if time.time() < deadline - 30:
        routing, max_load = _local_search_3bit(best_routing, max_iters=10000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Final local search polish (2-bit then 3-bit)
    if time.time() < deadline - 10:
        routing, max_load = _local_search_2bit(best_routing, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    if time.time() < deadline - 5:
        routing, max_load = _local_search_3bit(best_routing, max_iters=5000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    return best_routing
=======
    # Start with all clockwise
    best_routing = np.zeros(N_DEMANDS)
    best_max = _max_load(best_routing)
    
    rng = np.random.default_rng(42)
    
    # Method 1: Fast 2-bit local search from all-zero
    routing, max_load = _fast_local_search_2bit(np.zeros(N_DEMANDS), max_iters=100000)
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 2: Fast 2-bit local search from all-one
    routing, max_load = _fast_local_search_2bit(np.ones(N_DEMANDS), max_iters=100000)
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 3: Many random restarts with fast 2-bit local search
    n_restarts = 500
    for trial in range(n_restarts):
        if time.time() > deadline - 120:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        routing, max_load = _fast_local_search_2bit(random_routing, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: Fast SA from best found so far
    if time.time() < deadline - 100:
        routing, max_load = _fast_sa(best_routing, start_temp=5.0, end_temp=0.01, max_iters=100000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: Fast tabu search from best found so far
    if time.time() < deadline - 80:
        routing, max_load = _fast_tabu(best_routing, tabu_len=20, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 6: More random restarts with SA + tabu
    for trial in range(100):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        hc_routing, hc_max = _fast_local_search_2bit(random_routing, max_iters=30000)
        sa_routing, sa_max = _fast_sa(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=50000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
        
        if time.time() < deadline - 30:
            ts_routing, ts_max = _fast_tabu(sa_routing, tabu_len=15, max_iters=30000)
            if ts_max < best_max:
                best_max = ts_max
                best_routing = ts_routing.copy()
    
    # Final polish: fast 2-bit local search
    if time.time() < deadline - 10:
        routing, max_load = _fast_local_search_2bit(best_routing, max_iters=200000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    return best_routing
>>>>>>> REPLACE
