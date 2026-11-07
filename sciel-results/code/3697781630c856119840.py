# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Greedy + Local Search + Time-limited LP Refinement.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin

    # --- Step 1: Greedy Initialization ---
    fac_order = np.argsort(assign_cost, axis=1)
    load = np.zeros(n_fac, dtype=np.int64)
    result = np.full(n_cust, -1, dtype=np.int64)
    order = np.argsort(-demands)
    
    for i in order:
        assigned = False
        for f in fac_order[i]:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Fallback: find any facility with space
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
        if not assigned:
            # Should not happen if total capacity >= total demand
            # Force assign to facility 0 (might be invalid but _force_valid will fix)
            result[i] = 0
            load[0] += demands[i]

    # Ensure result is valid before local search
    result = _force_valid(result, demands, capacities, assign_cost)
    # Recalculate load for local search
    load = np.zeros(n_fac, dtype=np.int64)
    for i, f in enumerate(result):
        load[f] += demands[i]

    # --- Step 2: Local Search ---
    if time.time() < deadline - 30:
        result, load = _local_search_repair(result, load, demands, capacities, open_cost, assign_cost, deadline)

    # --- Step 3: LP Refinement (if time remains) ---
    # We try to solve the LP relaxation to get a better starting point or direct improvement.
    # However, solving the full LP is expensive. 
    # Instead, we check if we have enough time for a quick LP solve.
    # If LP solves successfully and quickly, we use its rounded solution as a new start for local search.
    
    if time.time() < deadline - 20:
        try:
            # Build LP matrices
            c = np.zeros(n_cust * n_fac + n_fac)
            c[:n_cust * n_fac] = assign_cost.flatten()
            c[n_cust * n_fac:] = open_cost

            n_rows = n_fac + n_cust * n_fac
            A_ub = np.zeros((n_rows, n_cust * n_fac + n_fac))
            b_ub = np.zeros(n_rows)
            
            # Capacity Constraints
            for j in range(n_fac):
                for i in range(n_cust):
                    idx = i * n_fac + j
                    A_ub[j, idx] = demands[i]
                A_ub[j, n_cust * n_fac + j] = -capacities[j]
                b_ub[j] = 0.0

            # Linking Constraints
            row_idx = n_fac
            for i in range(n_cust):
                for j in range(n_fac):
                    idx_x = i * n_fac + j
                    idx_y = n_cust * n_fac + j
                    A_ub[row_idx, idx_x] = 1.0
                    A_ub[row_idx, idx_y] = -1.0
                    b_ub[row_idx] = 0.0
                    row_idx += 1

            # Equality Constraints
            A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
            b_eq = np.ones(n_cust)
            for i in range(n_cust):
                for j in range(n_fac):
                    idx = i * n_fac + j
                    A_eq[i, idx] = 1.0

            bounds = [(0, 1)] * (n_cust * n_fac + n_fac)

            lp_start = time.time()
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            lp_elapsed = time.time() - lp_start

            if res.success and lp_elapsed < 10: # Only use if fast
                x_lp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Rounding
                result_lp = np.argmax(x_lp, axis=1).astype(np.int64)
                
                # Check validity
                load_lp = np.zeros(n_fac, dtype=np.int64)
                for i, f in enumerate(result_lp):
                    load_lp[f] += demands[i]
                
                if not np.any(load_lp > capacities):
                    # Compare objective
                    obj_current = _compute_obj(result, demands, capacities, open_cost, assign_cost)
                    obj_lp = _compute_obj(result_lp, demands, capacities, open_cost, assign_cost)
                    
                    if obj_lp < obj_current:
                        result = result_lp
                        load = load_lp
                        
                        # Refine with local search
                        if time.time() < deadline - 5:
                            result, load = _local_search_repair(result, load, demands, capacities, open_cost, assign_cost, deadline)

        except Exception:
            pass

    return result.astype(int)

def _compute_obj(result, demands, capacities, open_cost, assign_cost):
    n_cust = len(result)
    n_fac = len(capacities)
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

def _local_search_repair(result, load, demands, capacities, open_cost, assign_cost, deadline):
    """
    Local search that ensures validity and minimizes cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = load.copy()
    
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
                
    # Main local search for cost improvement
    max_iters = 500
    iters = 0
    improved = True
    while improved and iters < max_iters and time.time() < deadline - 5:
        improved = False
        iters += 1
        
        # Vectorized move evaluation
        cur_assign_cost = np.array([assign_cost[i, result[i]] for i in range(n_cust)])
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
        if not np.any(improving):
            break
        
        improving_indices = np.where(improving)[0]
        improving_indices = improving_indices[np.argsort(best_delta[improving_indices])]
        
        modified_fac = np.zeros(n_fac, dtype=bool)
        modified_cust = np.zeros(n_cust, dtype=bool)
        
        for i in improving_indices:
            if time.time() > deadline - 5:
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
            improved = True
            
    return result, load

# EVOLVE-BLOCK-END
