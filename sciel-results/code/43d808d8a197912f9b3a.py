# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin
    
    # Precompute sorted facility indices by assignment cost for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
    def _compute_obj(result):
        """Compute objective for a valid assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    def _greedy(order):
        """Greedy assign in given order, cheapest feasible facility first."""
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                return None
        return result
    
    # Try multiple orderings
    best_result = None
    best_cost = float('inf')
    
    # Ordering 1: largest demand first
    order1 = np.argsort(-demands)
    r = _greedy(order1)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Ordering 2: smallest demand first
    order2 = np.argsort(demands)
    r = _greedy(order2)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Ordering 3: random
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline:
            break
        order = rng.permutation(n_cust)
        r = _greedy(order)
        if r is not None:
            c = _compute_obj(r)
            if c < best_cost:
                best_cost = c
                best_result = r.copy()
    
    if best_result is None:
        # Fallback: round-robin
        result = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        valid = True
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities):
            # Just return something; will be invalid but we need to return something
            return result
        return result
    
    # Local search: vectorized move evaluation
    def _local_search(result, max_iters=200):
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        current_obj = _compute_obj(result)
        
        # Precompute: for each customer, the assignment cost to each facility
        # and the current facility's assignment cost
        cur_assign_cost = np.array([assign_cost[i, result[i]] for i in range(n_cust)])
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # For each customer, find the best facility to move to (vectorized)
            # delta_assign[i, f] = assign_cost[i, f] - assign_cost[i, result[i]]
            # But we need to handle open cost changes
            
            # Vectorized: compute all potential moves
            # For customer i moving from cur_f to f:
            #   delta = assign_cost[i,f] - assign_cost[i,cur_f] + open_cost_change
            #   open_cost_change = -open_cost[cur_f] if load[cur_f]==demands[i] else 0
            #                    + open_cost[f] if load[f]==0 else 0
            
            # Build delta matrix: (n_cust, n_fac)
            # Base delta from assignment cost
            delta_base = assign_cost - cur_assign_cost[:, np.newaxis]
            
            # Open cost adjustment:
            # If customer i is the only one at cur_f, closing it saves open_cost[cur_f]
            # If customer i is the first at f, opening it costs open_cost[f]
            
            # is_last_at_cur[i] = (load[result[i]] == demands[i])
            cur_fac = result
            is_last_at_cur = (load[cur_fac] == demands)
            
            # is_first_at_f[i, f] = (load[f] == 0) and f != cur_fac[i]
            # But we only care about f where load[f] == 0
            zero_load_fac = (load == 0)
            
            # For each customer i and target facility f:
            # open_delta = -open_cost[cur_fac[i]] if is_last_at_cur[i] and f != cur_fac[i] else 0
            #            + open_cost[f] if zero_load_fac[f] and f != cur_fac[i] else 0
            
            # Vectorized computation
            # open_delta[i, f] = 
            #   (-open_cost[cur_fac[i]] * is_last_at_cur[i] * (f != cur_fac[i]))
            #   + (open_cost[f] * zero_load_fac[f] * (f != cur_fac[i]))
            
            # Create mask for f != cur_fac[i]
            not_cur = np.ones((n_cust, n_fac), dtype=bool)
            not_cur[np.arange(n_cust), cur_fac] = False
            
            # Term 1: save open cost if closing current facility
            term1 = np.where(is_last_at_cur[:, np.newaxis], -open_cost[cur_fac][:, np.newaxis], 0.0)
            term1 = np.where(not_cur, term1, 0.0)
            
            # Term 2: pay open cost if opening new facility
            term2 = np.where(zero_load_fac[np.newaxis, :], open_cost[np.newaxis, :], 0.0)
            term2 = np.where(not_cur, term2, 0.0)
            
            delta_open = term1 + term2
            total_delta = delta_base + delta_open
            
            # Mask out current facility and infeasible moves
            total_delta[np.arange(n_cust), cur_fac] = np.inf
            
            # Mask out infeasible: load[f] + demands[i] > capacities[f]
            infeasible = (load[np.newaxis, :] + demands[:, np.newaxis]) > capacities[np.newaxis, :]
            total_delta[infeasible] = np.inf
            
            # Find best move for each customer
            best_f = np.argmin(total_delta, axis=1)
            best_delta = np.min(total_delta, axis=1)
            
            # Apply all improving moves simultaneously (but check conflicts)
            # To avoid conflicts, process customers in order of improvement
            improving = best_delta < -1e-9
            if not np.any(improving):
                break
            
            # Sort by improvement magnitude (most negative first)
            improving_indices = np.where(improving)[0]
            improving_indices = improving_indices[np.argsort(best_delta[improving_indices])]
            
            # Track which facilities are being modified to avoid conflicts
            # Simple approach: process sequentially, skip if target is already modified
            modified_fac = np.zeros(n_fac, dtype=bool)
            modified_cust = np.zeros(n_cust, dtype=bool)
            
            for i in improving_indices:
                if time.time() > deadline:
                    break
                if modified_cust[i]:
                    continue
                f_new = best_f[i]
                f_old = result[i]
                if modified_fac[f_new] or modified_fac[f_old]:
                    continue
                
                # Check capacity again (load may have changed)
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                
                # Apply move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                modified_fac[f_old] = True
                modified_fac[f_new] = True
                modified_cust[i] = True
                improved = True
        
        return result
    
    best_result = _local_search(best_result, max_iters=150)
    best_cost = _compute_obj(best_result)
    
    # Pair swap search with vectorized evaluation
    def _pair_swap_search(result, max_rounds=500):
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        current_obj = _compute_obj(result)
        rng_swap = np.random.RandomState(99)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            # Pick two random customers
            i, j = rng_swap.choice(n_cust, size=2, replace=False)
            fi, fj = result[i], result[j]
            if fi == fj:
                continue
            # Check if swap is feasible
            new_load_fi = load[fi] - demands[i] + demands[j]
            new_load_fj = load[fj] - demands[j] + demands[i]
            if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                continue
            
            # Change in assignment cost
            delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
            
            # Change in open cost
            delta_open = 0.0
            if new_load_fi == 0 and load[fi] > 0:
                delta_open -= open_cost[fi]
            if new_load_fi > 0 and load[fi] == 0:
                delta_open += open_cost[fi]
            if new_load_fj == 0 and load[fj] > 0:
                delta_open -= open_cost[fj]
            if new_load_fj > 0 and load[fj] == 0:
                delta_open += open_cost[fj]
            
            total_delta = delta_assign + delta_open
            if total_delta < -1e-9:
                load[fi] = new_load_fi
                load[fj] = new_load_fj
                result[i] = fj
                result[j] = fi
                current_obj += total_delta
        
        return result
    
    best_result = _pair_swap_search(best_result, max_rounds=500)
    best_cost = _compute_obj(best_result)
    
    # Final polish: another local search pass
    if time.time() < deadline - 5:
        best_result = _local_search(best_result, max_iters=100)
        best_cost = _compute_obj(best_result)
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
