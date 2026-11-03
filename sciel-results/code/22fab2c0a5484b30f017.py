"""Improved QAP solver for tai256c using greedy construction + fast 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP objective: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return np.dot(a.ravel(), bp.ravel())


def _greedy_initial(a, b, n, rng):
    """
    Greedy construction: pick facilities one by one, placing each at the location
    that minimizes the incremental cost given already-placed facilities.
    Start with the facility with the largest total flow.
    """
    # Precompute row sums of a to order facilities by importance
    a_row_sums = a.sum(axis=1)
    # We'll process facilities in order of decreasing total flow (heavy first)
    # But to allow some diversity, we can shuffle ties or use a random tie-break
    order = np.argsort(-a_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    p_facilities = np.empty(n, dtype=np.int64)  # p_facilities[k] = facility placed at step k
    p_locations = np.empty(n, dtype=np.int64)   # p_locations[k] = location assigned at step k
    
    # Track incremental cost contributions
    # When we place facility f at location l, the incremental cost is:
    # sum over already placed facilities g: a[f,g]*b[l, loc_g] + a[g,f]*b[loc_g, l]
    # Since a is symmetric in QAP (typically), a[f,g]=a[g,f], but let's be general.
    
    # Actually for QAP, the cost is sum_{i,j} a[i,j] b[p[i],p[j]].
    # When placing facility f at location l, the new terms involving f are:
    # sum_{j already placed} a[f,j] * b[l, p[j]] + sum_{i already placed} a[i,f] * b[p[i], l]
    # + a[f,f] * b[l,l] (but b[l,l] is typically 0 for distance matrices)
    
    # Let's use a simpler approach: maintain a "cost contribution" array for each location.
    # cost_at_loc[l] = incremental cost if we place the next facility at location l.
    
    cost_at_loc = np.zeros(n, dtype=np.float64)
    
    for step in range(n):
        f = order[step]
        # Find the location that minimizes cost_at_loc
        # But cost_at_loc needs to be updated based on which facilities are already placed.
        
        # Actually, let's recompute: for the current facility f, the incremental cost
        # of placing it at location l is:
        # sum_{g: already placed} (a[f,g] * b[l, p[g]] + a[g,f] * b[p[g], l])
        
        # This is expensive to compute naively. Let's maintain it incrementally.
        
        # Better approach: maintain for each location l, the accumulated cost contribution.
        # When facility g is placed at location pg, it adds to location l:
        # a[g,l_facility] * b[pg, l] + a[l_facility, g] * b[l, pg]
        # This is getting complex. Let's just do a simpler greedy: 
        # for each step, evaluate all unplaced locations for the current facility.
        
        if step == 0:
            # Place first facility at the "central" location (min row sum of b)
            b_row_sums = b.sum(axis=1)
            best_loc = np.argmin(b_row_sums)
        else:
            # Evaluate cost for each unplaced location
            best_cost = np.inf
            best_loc = -1
            for l in range(n):
                if placed[l]:
                    continue
                # Compute incremental cost of placing f at l
                # Sum over already placed facilities g:
                # a[f,g]*b[l, p[g]] + a[g,f]*b[p[g], l]
                inc_cost = 0.0
                for k in range(step):
                    g = p_facilities[k]
                    pg = p_locations[k]
                    inc_cost += a[f, g] * b[l, pg] + a[g, f] * b[pg, l]
                if inc_cost < best_cost:
                    best_cost = inc_cost
                    best_loc = l
        
        placed[best_loc] = True
        p_facilities[step] = f
        p_locations[step] = best_loc
        p[f] = best_loc
    
    return p


def _local_search_2opt(a, b, p, time_limit=110.0):
    """
    2-opt local search: try swapping two facilities' locations.
    Use vectorized cost difference computation.
    """
    n = len(p)
    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    start_time = time.time()
    
    # Precompute: for each pair (i,j) of facilities, the cost difference of swapping their locations.
    # If facility i is at location pi and facility j is at location pj, swapping gives:
    # delta = sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) + sum_k a[k,i]*(b[pk,pj] - b[pk,pi])
    #        + sum_k a[j,k]*(b[pi,pk] - b[pj,pk]) + sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
    #        + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    # Simplifying:
    # Let dp = b[:, pj] - b[:, pi]  (column difference)
    # delta = a[i,:] @ dp + a[:,i] @ dp + a[j,:] @ (-dp) + a[:,j] @ (-dp) + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #        = (a[i,:] - a[j,:]) @ dp + (a[:,i] - a[:,j]) @ dp + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #        = (a[i,:] - a[j,:] + a[:,i] - a[:,j]) @ dp + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    
    # Let's define S = a + a.T  (symmetric part, if a is not symmetric this handles both directions)
    # Actually, let's just compute directly.
    
    no_improvement_streak = 0
    max_no_improvement = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try all pairs in random order
        indices = np.arange(n)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        
        for idx_i in range(n):
            i = indices[idx_i]
            pi = best_p[i]
            for idx_j in range(idx_i + 1, n):
                j = indices[idx_j]
                pj = best_p[j]
                
                # Compute cost difference of swapping i and j
                # dp = b[:, pj] - b[:, pi]
                dp = b[:, pj] - b[:, pi]
                # delta = (a[i,:] - a[j,:]) @ dp + (a[:,i] - a[:,j]) @ dp + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                delta = (a[i, :] - a[j, :]) @ dp + (a[:, i] - a[:, j]) @ dp
                delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
                
                if delta < -1e-9:
                    # Perform swap
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
                    no_improvement_streak = 0
                    # Restart the scan from the beginning (or continue, but restart is safer)
                    break
            if improved:
                continue
            if time.time() - start_time > time_limit:
                break
        
        if not improved:
            no_improvement_streak += 1
            if no_improvement_streak >= max_no_improvement:
                break
        else:
            no_improvement_streak = 0
    
    return best_p, best_cost


def _ils(a, b, p, time_limit=110.0):
    """
    Iterated Local Search: perturb the current solution, then local search.
    Keep the best solution found.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Perturb: swap a random number of pairs
        cur_p = best_p.copy()
        num_swaps = np.random.randint(2, max(3, n // 10))
        for _ in range(num_swaps):
            i, j = np.random.choice(n, size=2, replace=False)
            cur_p[i], cur_p[j] = cur_p[j], cur_p[i]
        
        # Local search on perturbed solution
        cur_p, cur_cost = _local_search_2opt(a, b, cur_p, time_limit=max(1.0, time_limit - (time.time() - start_time)))
        
        if cur_cost < best_cost:
            best_p = cur_p.copy()
            best_cost = cur_cost
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    # Phase 1: Generate initial solutions using greedy construction with different orderings
    # Phase 2: Local search on each
    # Phase 3: ILS from the best
    
    # Generate a few diverse initial solutions
    candidates = []
    
    # Greedy with row-sum ordering
    p1 = _greedy_initial(a, b, n, np.random.default_rng(42))
    candidates.append(p1)
    
    # Greedy with reverse ordering
    p2 = _greedy_initial(a, b, n, np.random.default_rng(123))
    # Modify order: use ascending instead of descending
    # Actually _greedy_initial uses descending. Let's just use a different seed.
    candidates.append(p2)
    
    # Random initial
    p3 = np.random.default_rng(7).permutation(n)
    candidates.append(p3)
    
    # Rank-matching initial (from seed)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p4 = np.empty(n, dtype=np.int64)
    p4[heavy] = central
    candidates.append(p4)
    
    # Local search on each candidate
    best_p = None
    best_cost = np.inf
    
    for p in candidates:
        if time.time() - start_time > time_limit - 30:
            break
        p_ls, cost = _local_search_2opt(a, b, p, time_limit=max(5.0, time_limit - (time.time() - start_time)))
        if cost < best_cost:
            best_cost = cost
            best_p = p_ls
    
    # ILS from the best
    if time.time() - start_time < time_limit - 10:
        remaining = time_limit - (time.time() - start_time)
        best_p, best_cost = _ils(a, b, best_p, time_limit=remaining)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
