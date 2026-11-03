"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    Uses vectorized operations for speed.
    """
    u = p[i]
    v = p[j]
    
    # Direct terms
    old_ii = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_ii = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    # Cross terms with all other positions
    n = len(p)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) > 0:
        pk = p[k_indices]
        
        a_i_k = a[i, k_indices]
        a_j_k = a[j, k_indices]
        a_k_i = a[k_indices, i]
        a_k_j = a[k_indices, j]
        
        b_u_pk = b[u, pk]
        b_v_pk = b[v, pk]
        b_pk_u = b[pk, u]
        b_pk_v = b[pk, v]
        
        old_cross = np.sum(a_i_k * b_u_pk + a_j_k * b_v_pk + a_k_i * b_pk_u + a_k_j * b_pk_v)
        new_cross = np.sum(a_i_k * b_v_pk + a_j_k * b_u_pk + a_k_i * b_pk_v + a_k_j * b_pk_u)
    else:
        old_cross = 0
        new_cross = 0
    
    delta = (new_ii + new_cross) - (old_ii + old_cross)
    return int(delta)


def _two_opt_search(a, b, p, time_limit, start_time):
    """
    Perform 2-opt local search with first-improvement strategy.
    Uses a time-bounded loop with early termination.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break  # First improvement: restart scan from beginning
        
        if not improved:
            break
    
    return best_p, best_cost


def _greedy_init(a, b):
    """
    Greedy initialization: assign facilities to locations one by one
    choosing the assignment that minimizes incremental cost.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    # Precompute row sums for ordering
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Start with the facility having highest flow demand
    order = np.argsort(-a_row_sums)
    
    # For each facility in order, pick the location that minimizes incremental cost
    for idx in order:
        best_loc = -1
        best_cost = float('inf')
        
        for loc in range(n):
            if assigned[loc]:
                continue
            
            # Compute incremental cost of assigning facility idx to location loc
            inc_cost = 0
            for j in range(n):
                if assigned[j]:
                    # j is already assigned to location p[j]
                    inc_cost += a[idx, j] * b[loc, p[j]]
                    inc_cost += a[j, idx] * b[p[j], loc]
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        p[idx] = best_loc
        assigned[best_loc] = True
    
    return p


def _rank_matching_init(a, b):
    """
    Rank-matching heuristic: match high-flow facilities to low-distance locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: leave 15s margin for safety (150s total, use 135s)
    time_budget = 135
    
    best_p = None
    best_cost = float('inf')
    
    # Strategy 1: Rank-matching initialization + 2-opt
    p_rm = _rank_matching_init(a, b)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        p_opt, cost_opt = _two_opt_search(a, b, p_rm, time_limit=remaining - 5, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Strategy 2: Greedy initialization + 2-opt
    p_greedy = _greedy_init(a, b)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        p_opt, cost_opt = _two_opt_search(a, b, p_greedy, time_limit=remaining - 5, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Strategy 3: Random restarts with 2-opt
    rng = np.random.default_rng(12345)
    restart_count = 0
    while time.time() - start_time < time_budget - 20:
        restart_count += 1
        p_rand = rng.permutation(n)
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 15:
            break
        
        # Shorter time limit per restart
        per_restart_time = min(remaining - 5, 20)
        if per_restart_time < 5:
            break
        
        p_opt, cost_opt = _two_opt_search(a, b, p_rand, time_limit=per_restart_time, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Final safety check
    if time.time() - start_time > time_budget:
        # Shouldn't happen, but just in case
        pass
    
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
