"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return int(np.sum(a * b_p))

def _init_rank_matching(a, b):
    """Rank-matching initialization: sort facilities by flow centrality, locations by distance centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    dist_centrality = b.sum(axis=0) + b.sum(axis=1)
    
    heavy = np.argsort(-flow_centrality, kind="stable")
    central = np.argsort(dist_centrality, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _init_greedy(a, b):
    """Greedy construction: place facilities one by one in order of flow centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    order = np.argsort(-flow_centrality, kind="stable")
    
    p = np.full(n, -1, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used_locs[loc]:
                continue
            placed_mask = p >= 0
            if not np.any(placed_mask):
                cost = 0
            else:
                placed_facs = np.where(placed_mask)[0]
                cost = np.dot(a[fac, placed_facs], b[loc, p[placed_facs]]) + \
                       np.dot(a[placed_facs, fac], b[p[placed_facs], loc])
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used_locs[best_loc] = True
    
    return p

def _swap_delta(a, b, p, i, j):
    """
    Fast swap delta computation using full vectors minus i,j contributions.
    """
    li = p[i]
    lj = p[j]
    
    # Full dot products
    diff1 = b[lj, p] - b[li, p]
    t1_full = np.dot(a[i, :], diff1)
    
    diff2 = b[p, lj] - b[p, li]
    t2_full = np.dot(a[:, i], diff2)
    
    diff3 = b[li, p] - b[lj, p]
    t3_full = np.dot(a[j, :], diff3)
    
    diff4 = b[p, li] - b[p, lj]
    t4_full = np.dot(a[:, j], diff4)
    
    # Subtract i and j contributions
    t1_i = a[i, i] * (b[lj, li] - b[li, li])
    t1_j = a[i, j] * (b[lj, lj] - b[li, lj])
    t1 = t1_full - t1_i - t1_j
    
    t2_i = a[i, i] * (b[li, lj] - b[li, li])
    t2_j = a[j, i] * (b[lj, lj] - b[lj, li])
    t2 = t2_full - t2_i - t2_j
    
    t3_i = a[j, i] * (b[li, li] - b[lj, li])
    t3_j = a[j, j] * (b[li, lj] - b[lj, lj])
    t3 = t3_full - t3_i - t3_j
    
    t4_i = a[i, j] * (b[li, li] - b[lj, li])
    t4_j = a[j, j] * (b[lj, lj] - b[lj, lj])
    t4 = t4_full - t4_i - t4_j
    
    t5 = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    t6 = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
    
    return int(t1 + t2 + t3 + t4 + t5 + t6)

def _sa_run(a, b, p_init, max_time, rng, t_start, t_end, cooling_rate):
    """Run simulated annealing for max_time seconds, return best permutation and cost."""
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    steps = 0
    accepted = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        if remaining := (max_time - elapsed) < 0.05:
            break
        
        # Process in small batches with time check
        batch = 5000
        for _ in range(batch):
            if time.time() - start_time > max_time:
                break
            
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            if delta <= 0:
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                accepted += 1
            else:
                if t_start > t_end:
                    # Accept with probability exp(-delta/t)
                    # Use log for numerical stability
                    if delta / t_start < 700:
                        if math.log(rng.random()) < -delta / t_start:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            accepted += 1
            
            # Cool
            t_start *= cooling_rate
            if t_start < t_end:
                t_start = t_end
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        steps += batch
    
    return best_p, best_cost

def _hill_climb(a, b, p, max_time):
    """First-improvement hill climbing with 2-opt swaps."""
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    start = time.time()
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for i in range(n):
            if time.time() - start > max_time:
                break
            for j in range(i + 1, n):
                if time.time() - start > max_time:
                    break
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    max_total_time = 115.0
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Run 1: Rank matching + long SA
    p_init = _init_rank_matching(a, b)
    p_result, cost_result = _sa_run(a, b, p_init, 30.0, rng, 50000.0, 1.0, 0.99992)
    if cost_result < best_cost:
        best_cost = cost_result
        best_p = p_result
    
    # Run 2: Perturb best, shorter SA
    if time.time() - start_time < max_total_time - 25.0:
        p_init = best_p.copy()
        n_swaps = n // 6
        for _ in range(n_swaps):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            p_init[i], p_init[j] = p_init[j], p_init[i]
        p_result, cost_result = _sa_run(a, b, p_init, 22.0, rng, 20000.0, 1.0, 0.99995)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Run 3: Greedy init + SA
    if time.time() - start_time < max_total_time - 20.0:
        p_init = _init_greedy(a, b)
        p_result, cost_result = _sa_run(a, b, p_init, 18.0, rng, 30000.0, 1.0, 0.99993)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Run 4: Another perturbation from best
    if time.time() - start_time < max_total_time - 12.0:
        p_init = best_p.copy()
        n_swaps = n // 4
        for _ in range(n_swaps):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            p_init[i], p_init[j] = p_init[j], p_init[i]
        p_result, cost_result = _sa_run(a, b, p_init, 10.0, rng, 10000.0, 1.0, 0.99997)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Final hill climb
    if best_p is not None and time.time() - start_time < max_total_time - 1.0:
        remaining = max_total_time - 1.0 - (time.time() - start_time)
        if remaining > 2.0:
            p_result, cost_result = _hill_climb(a, b, best_p, remaining)
            if cost_result < best_cost:
                best_cost = cost_result
                best_p = p_result
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
