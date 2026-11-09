<<<<<<< SEARCH
def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using a combination of
    local search, simulated annealing, genetic algorithm, and
    smooth surrogate optimization.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    best_routing = None
    best_max = float('inf')
    
    # Phase 0a: Try MILP for exact global optimum (fast for this small problem)
    if time.time() < deadline - 5:
        mr, mm = _milp_solve(cw_edges, ccw_edges, demands, deadline - 5)
        if mr is not None and mm < best_max:
            best_max = mm
            best_routing = mr.copy()
    
    # Phase 0b: Brute force all 2^15 = 32768 combinations (feasible!)
    # This guarantees the global optimum
    br, bm = _brute_force_small()
    if bm < best_max:
        best_max = bm
        best_routing = br.copy()
    
    # Phase 1: Multiple random restarts with local search (to verify/improve)
    np_random = np.random.default_rng(2024)
    
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        
        # Random initial solution
        routing = np_random.random(D)
        
        # Local search
        lr, lm = _local_search(routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=50000)
        if lm < best_max - 1e-9:
            best_max = lm
            best_routing = lr.copy()
    
    # Phase 2: Pair local search from best
    if best_routing is not None and time.time() < deadline - 20:
        pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 10, max_iter=20000)
        if pm < best_max - 1e-9:
            best_max = pm
            best_routing = pr.copy()
    
    # Phase 3: Simulated annealing from best
    if best_routing is not None and time.time() < deadline - 10:
        sr, sm = _simulated_annealing(best_routing, cw_edges, ccw_edges, demands, deadline - 5, T0=1.0, cooling=0.9998, max_iter=100000)
        if sm < best_max - 1e-9:
            best_max = sm
            best_routing = sr.copy()
    
    # Phase 4: Smooth gradient descent to find good continuous solution
    if time.time() < deadline - 5:
        # Try from best binary solution
        gr, gm = _smooth_gradient_descent(best_routing, cw_edges, ccw_edges, demands, deadline - 2, beta=5.0, lr=0.01, max_iter=2000)
        if gm < best_max - 1e-9:
            best_max = gm
            best_routing = gr.copy()
        
        # Try from random
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            rand_routing = np_random.random(D)
            gr, gm = _smooth_gradient_descent(rand_routing, cw_edges, ccw_edges, demands, deadline - 2, beta=5.0, lr=0.01, max_iter=1000)
            if gm < best_max - 1e-9:
                best_max = gm
                best_routing = gr.copy()
    
    # Phase 5: Genetic algorithm
    if time.time() < deadline - 5:
        ga_r, ga_m = _genetic_algorithm(cw_edges, ccw_edges, demands, deadline - 2, pop_size=40, max_gen=200)
        if ga_m < best_max - 1e-9:
            best_max = ga_m
            best_routing = ga_r.copy()
    
    # Final polish: local search + pair search on the best found
    if best_routing is not None:
        if time.time() < deadline - 2:
            fr, fm = _local_search(best_routing, cw_edges, ccw_edges, demands, deadline - 1, max_iter=100000)
            if fm < best_max - 1e-9:
                best_max = fm
                best_routing = fr.copy()
        
        if time.time() < deadline - 1:
            pr, pm = _pair_local_search(best_routing, cw_edges, ccw_edges, demands, deadline, max_iter=10000)
            if pm < best_max - 1e-9:
                best_max = pm
                best_routing = pr.copy()
    
    # Ensure we have a valid routing
    if best_routing is None:
        best_routing = np.zeros(D)
    
    return best_routing
=======
def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using vectorized brute force
    to find the globally optimal routing.
    """
    cw_edges, ccw_edges, demands = _build_routes()
    
    # Vectorized brute force finds the exact global optimum
    # for this problem size (15 demands) in milliseconds.
    best_routing, best_max = _brute_force_small(cw_edges, ccw_edges, demands)
    
    return best_routing
>>>>>>> REPLACE
