# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: MILP (if available and fast) -> Greedy + Local Search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin

    # Try MILP first
    result = _try_milp(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is None:
        # Fallback to Greedy + Local Search
        result = _greedy_init(demands, capacities, assign_cost)
        if time.time() < deadline - 10:
            result = _local_search_repair(result, demands, capacities, open_cost, assign_cost, deadline)
        else:
            result = _force_valid(result, demands, capacities, assign_cost)
            
    return result.astype(int)

def _try_milp(demands, capacities, open_cost, assign_cost, deadline):
    """
    Attempt to solve using scipy.optimize.milp.
    Returns None if not available, fails, or takes too long.
    """
    if not hasattr(opt, 'milp'):
        return None
        
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables: x (n_cust * n_fac), y (n_fac)
    n_vars = n_cust * n_fac + n_fac
    
    # Objective
    c = np.zeros(n_vars)
    c[:n_cust * n_fac] = assign_cost.flatten()
    c[n_cust * n_fac:] = open_cost
    
    # Constraints
    # 1. Sum_j x_ij = 1 for each i
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
            
    # 2. Capacity: Sum_i d_i x_ij - C_j y_j <= 0
    # 3. Linking: x_ij - y_j <= 0
    # Combine into A_ub
    n_ub_rows = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_ub_rows, n_vars))
    b_ub = np.zeros(n_ub_rows)
    
    for j in range(n_fac):
        # Capacity constraint
        for i in range(n_cust):
            A_ub[j, i * n_fac + j] = demands[i]
        A_ub[j, n_cust * n_fac + j] = -capacities[j]
        b_ub[j] = 0.0
        
    for i in range(n_cust):
        for j in range(n_fac):
            row = n_fac + i * n_fac + j
            A_ub[row, i * n_fac + j] = 1.0
            A_ub[row, n_cust * n_fac + j] = -1.0
            b_ub[row] = 0.0
            
    # Bounds
    bounds = []
    for _ in range(n_cust * n_fac):
        bounds.append((0, 1))
    for _ in range(n_fac):
        bounds.append((0, 1))
        
    # Integrality
    integrality = np.ones(n_vars)
    
    try:
        start_time = time.time()
        res = opt.milp(c, constraints=[opt.LinearConstraint(A_ub, ub=b_ub), opt.LinearConstraint(A_eq, lb=b_eq, ub=b_eq)], bounds=bounds, integrality=integrality, options={"time_limit": 30})
        
        if time.time() > deadline - 10:
            return None
            
        if res.status == 0: # Optimal
            x = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Rounding is exact for binary
            result = np.argmax(x, axis=1).astype(int)
            return result
        else:
            return None
    except Exception:
        return None

def _greedy_init(demands, capacities, assign_cost):
    """
    Greedy initialization: Sort customers by demand descending, assign to cheapest feasible facility.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute facility preference order for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
    # Sort customers by demand descending
    cust_order = np.argsort(-demands)
    
    load = np.zeros(n_fac, dtype=np.int64)
    result = np.full(n_cust, -1, dtype=np.int64)
    
    for i in cust_order:
        assigned = False
        for f in fac_order[i]:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Fallback: find any feasible facility
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Should not happen if total capacity >= total demand
                # Force assign to least loaded
                f = np.argmin(load)
                load[f] += demands[i]
                result[i] = f
                
    return result

def _force_valid(result, demands, capacities, assign_cost):
    """
    Emergency repair: if result is invalid, make it valid by moving customers
    to any facility with space, prioritizing lowest assignment cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    
    # First pass: check validity
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
            
    # If any capacity exceeded, we need to redistribute.
    if np.any(load > capacities):
        # Reset and re-greedy
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        order = np.argsort(-demands)
        fac_order = np.argsort(assign_cost, axis=1)
        
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Find any facility with space
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                # Should not happen if total capacity >= total demand
                result[i] = 0
                load[0] += demands[i]
                
    return result

def _local_search_repair(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Local search that ensures validity and minimizes cost.
    Uses single-move + pair-swap moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    for i, f in enumerate(result):
        load[f] += demands[i]
        
    # Ensure valid start
    if np.any(load > capacities):
        # Fix violations first
        max_iters_fix = 1000
        for _ in range(max_iters_fix):
            if time.time() > deadline:
                break
            over_facs = np.where(load > capacities)[0]
            if len(over_facs) == 0:
                break
            f_over = over_facs[0]
            moved = False
            best_move_cost = float('inf')
            best_i = -1
            best_f_new = -1
            
            custs_in_f = np.where(result == f_over)[0]
            
            for i in custs_in_f:
                for f_new in range(n_fac):
                    if f_new == f_over:
                        continue
                    if load[f_new] + demands[i] <= capacities[f_new]:
                        cost = assign_cost[i, f_new]
                        if cost < best_move_cost:
                            best_move_cost = cost
                            best_i = i
                            best_f_new = f_new
            
            if best_i != -1:
                load[f_over] -= demands[best_i]
                load[best_f_new] += demands[best_i]
                result[best_i] = best_f_new
                moved = True
            else:
                break
                
    # Main local search
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > deadline - 3:
            break
        
        # --- Single move phase (vectorized) ---
        cur_assign_cost = assign_cost[np.arange(n_cust), result]
        delta_base = assign_cost - cur_assign_cost[:, np.newaxis]
        
        cur_fac = result
        is_last_at_cur = (load[cur_fac] == demands)
        zero_load_fac = (load == 0)
        
        not_cur = np.ones((n_cust, n_fac), dtype=bool)
        not_cur[np.arange(n_cust), cur_fac] = False
        
        term1 = np.where(is_last_at_cur[:, np.newaxis], -open_cost[cur_fac][:, np.newaxis], 0.0)
        term1 = np.where(not_cur, term1, 0.0)
        
        term2 = np.where(zero_load_fac[np.newaxis, :], open_cost[np.newaxis, :], 0.0)
        term2 = np.where(not_cur, term2, 0.0)
        
        delta_open = term1 + term2
        total_delta = delta_base + delta_open
        
        total_delta[np.arange(n_cust), cur_fac] = np.inf
        infeasible = (load[np.newaxis, :] + demands[:, np.newaxis]) > capacities[np.newaxis, :]
        total_delta[infeasible] = np.inf
        
        best_f = np.argmin(total_delta, axis=1)
        best_delta = np.min(total_delta, axis=1)
        
        improving = best_delta < -1e-9
        if np.any(improving):
            improving_indices = np.where(improving)[0]
            improving_indices = improving_indices[np.argsort(best_delta[improving_indices])]
            
            modified_fac = np.zeros(n_fac, dtype=bool)
            modified_cust = np.zeros(n_cust, dtype=bool)
            any_applied = False
            
            for i in improving_indices:
                if time.time() > deadline - 3:
                    break
                if modified_cust[i]:
                    continue
                f_new = best_f[i]
                f_old = result[i]
                if modified_fac[f_new] or modified_fac[f_old]:
                    continue
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                modified_fac[f_old] = True
                modified_fac[f_new] = True
                modified_cust[i] = True
                any_applied = True
            
            if any_applied:
                continue  # Restart single-move phase with updated state
        
        # --- Pair swap phase (Optimized) ---
        # Vectorized check for all pairs (i, j)
        f_of = result
        f_i = f_of[:, None]
        f_j = f_of[None, :]
        mask_diff_fac = f_i != f_j
        
        ac_i_fi = assign_cost[np.arange(n_cust), f_of]
        ac_j_fj = ac_i_fi # same values
        
        ac_j_fi = assign_cost[np.arange(n_cust)[None, :], f_i]
        ac_i_fj = assign_cost[np.arange(n_cust)[:, None], f_j]
        
        delta = ac_j_fi + ac_i_fj - ac_i_fi[:, None] - ac_j_fj[None, :]
        
        load_fi = load[f_i]
        load_fj = load[f_j]
        new_load_fi = load_fi - demands[:, None] + demands[None, :]
        new_load_fj = load_fj - demands[None, :] + demands[:, None]
        
        cap_ok = (new_load_fi <= capacities[f_i]) & (new_load_fj <= capacities[f_j])
        
        valid = mask_diff_fac & cap_ok & (delta < -1e-9)
        
        if np.any(valid):
            deltas = np.where(valid, delta, np.inf)
            min_idx = np.argmin(deltas)
            i, j = np.unravel_index(min_idx, deltas.shape)
            f_i_val = f_of[i]
            f_j_val = f_of[j]
            result[i] = f_j_val
            result[j] = f_i_val
            load[f_i_val] -= demands[i]
            load[f_i_val] += demands[j]
            load[f_j_val] -= demands[j]
            load[f_j_val] += demands[i]
            continue # Restart
        
        # If no improvement in either phase, stop
        break
        
    return result

# EVOLVE-BLOCK-END
