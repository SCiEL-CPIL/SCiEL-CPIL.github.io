<<<<<<< SEARCH
def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer under 120s limit
    
    rng = random.Random(42)
    
    # Initial solution: rank matching
    p = _rank_matching_init(a, b)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, best_p)
    
    # First local search
    current_p, current_cost = _local_search(a, b, p, best_cost, deadline)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost
    
    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() < deadline - 10:
        iteration += 1
        
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        # Perturb from best solution
        p = _perturb(best_p, k=perturb_k, rng=rng)
        cost = _compute_cost(a, b, p)
        
        # Local search from perturbed
        p, cost = _local_search(a, b, p, cost, deadline)
        
        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = _hungarian_reopt(a, b, p)
            h_cost = _compute_cost(a, b, h_p)
            h_p, h_cost = _local_search(a, b, h_p, h_cost, deadline)
            if h_cost < cost:
                p, cost = h_p, h_cost
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Final polish: one more local search pass on best solution
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = _local_search(a, b, final_p, final_cost, deadline)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()
=======
def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer under 120s limit
    
    rng = random.Random(42)
    
    # Initial solution: rank matching
    p = _rank_matching_init(a, b)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, best_p)
    
    # First local search
    current_p, current_cost = _local_search(a, b, p, best_cost, deadline)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost
    
    # Iterated Local Search with periodic Hungarian re-optimization and VNS
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3
    use_vns = False  # Flag to alternate between standard and VNS local search
    
    while time.time() < deadline - 10:
        iteration += 1
        
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        # Alternate between standard and VNS perturbation
        if iteration % 2 == 0:
            p = _perturb_vns(best_p, k=perturb_k, rng=rng)
        else:
            p = _perturb(best_p, k=perturb_k, rng=rng)
        
        cost = _compute_cost(a, b, p)
        
        # Alternate between standard and VNS local search
        if iteration % 2 == 0:
            p, cost = _vns_local_search(a, b, p, cost, deadline)
        else:
            p, cost = _local_search(a, b, p, cost, deadline)
        
        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = _hungarian_reopt(a, b, p)
            h_cost = _compute_cost(a, b, h_p)
            h_p, h_cost = _local_search(a, b, h_p, h_cost, deadline)
            if h_cost < cost:
                p, cost = h_p, h_cost
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Final polish: one more local search pass on best solution
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = _local_search(a, b, final_p, final_cost, deadline)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()
>>>>>>> REPLACE
