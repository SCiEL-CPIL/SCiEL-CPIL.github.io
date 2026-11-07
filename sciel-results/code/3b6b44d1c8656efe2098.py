# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: LP Relaxation + Rounding + Repair + Local Search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin

    # If n_cust or n_fac is very small, fall back to brute force or simple greedy
    # But 100x20 is standard.

    # --- Step 1: Solve LP Relaxation ---
    # Variables:
    # x[i, j] : 1 if customer i is assigned to facility j (0..1 in LP)
    # y[j]    : 1 if facility j is open (0..1 in LP)
    # Total variables: n_cust * n_fac + n_fac
    
    # Objective: Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i,j] * x[i,j])
    # c vector for linprog
    c = np.zeros(n_cust * n_fac + n_fac)
    # x variables come first
    c[:n_cust * n_fac] = assign_cost.flatten()
    # y variables come last
    c[n_cust * n_fac:] = open_cost

    # Inequality Constraints: A_ub @ x <= b_ub
    # 1. Capacity constraints: sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    #    For each facility j:
    #    sum_i demands[i] * x[i,j] <= capacities[j] * y[j]
    #    sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    
    # 2. Linking constraints: x[i,j] <= y[j]
    #    x[i,j] - y[j] <= 0
    
    # We can combine these into A_ub.
    # Number of rows in A_ub:
    # n_fac (capacity) + n_cust * n_fac (linking)
    n_rows = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_rows, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_rows)
    
    # Fill Capacity Constraints (first n_fac rows)
    for j in range(n_fac):
        # x[i,j] coefficients
        col_start = j
        col_step = n_fac
        for i in range(n_cust):
            idx = i * n_fac + j
            A_ub[j, idx] = demands[i]
        # y[j] coefficient
        A_ub[j, n_cust * n_fac + j] = -capacities[j]
        b_ub[j] = 0.0

    # Fill Linking Constraints (next n_cust * n_fac rows)
    # x[i,j] - y[j] <= 0
    row_idx = n_fac
    for i in range(n_cust):
        for j in range(n_fac):
            idx_x = i * n_fac + j
            idx_y = n_cust * n_fac + j
            A_ub[row_idx, idx_x] = 1.0
            A_ub[row_idx, idx_y] = -1.0
            b_ub[row_idx] = 0.0
            row_idx += 1

    # Equality Constraints: A_eq @ x = b_eq
    # sum_j x[i,j] = 1 for each customer i
    n_eq_rows = n_cust
    A_eq = np.zeros((n_eq_rows, n_cust * n_fac + n_fac))
    b_eq = np.ones(n_eq_rows)
    
    for i in range(n_cust):
        for j in range(n_fac):
            idx = i * n_fac + j
            A_eq[i, idx] = 1.0

    # Bounds
    # x[i,j] in [0, 1]
    # y[j] in [0, 1]
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)

    # Solve LP with time guard
    x_lp = None
    try:
        lp_start = time.time()
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        lp_elapsed = time.time() - lp_start
        
        if res.success and lp_elapsed < 30:
            x_lp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
        else:
            x_lp = None
            
    except Exception:
        x_lp = None

    if x_lp is not None:
        # --- Step 2: Rounding with capacity-aware selection ---
        # Instead of simple argmax, do a greedy rounding that respects capacities
        # Sort customers by their LP "confidence" (max x value) descending
        # Assign each to best facility with remaining capacity
        
        result = np.full(n_cust, -1, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        
        # Compute confidence scores
        conf = x_lp.max(axis=1)
        order = np.argsort(-conf)
        
        for i in order:
            # Try facilities in order of LP value
            fac_order_i = np.argsort(-x_lp[i])
            assigned = False
            for f in fac_order_i:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                # Fallback: any facility with space
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
            if not assigned:
                # Should not happen
                result[i] = 0
                load[0] += demands[i]
        
        # Check if we are within time limit
        if time.time() < deadline - 10:
            result = _local_search_repair(result, demands, capacities, open_cost, assign_cost, deadline)
        else:
            result = _force_valid(result, demands, capacities, assign_cost)
    else:
        # Fallback: Greedy + Local Search (same as original but maybe faster start)
        fac_order = np.argsort(assign_cost, axis=1)
        def _greedy(order):
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
        
        order1 = np.argsort(-demands)
        r = _greedy(order1)
        if r is None:
            r = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
            
        if time.time() < deadline - 10:
            r = _local_search_repair(r, demands, capacities, open_cost, assign_cost, deadline)
        else:
            r = _force_valid(r, demands, capacities, assign_cost)
        result = r

    return result.astype(int)

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
    # Simple strategy: sort customers by demand descending, reassign all.
    # This is a bit heavy but ensures validity.
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
    Uses single-move + pair-swap moves with vectorized evaluation.
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
                
    # Main local search: alternate single-move and pair-swap
    max_rounds = 150
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
        
        # --- Pair swap phase ---
        # For each pair of customers (i, j) at different facilities, check if swapping helps
        # Vectorized: for each pair (i,j), delta = (ac[j,f_i]+ac[i,f_j]) - (ac[i,f_i]+ac[j,f_j])
        # Must respect capacity: after swap, loads must be valid
        
        # Build facility-of-customer array
        fac_of = result
        
        # Only consider pairs at different facilities
        # To keep it fast, sample a subset of pairs or do full O(n^2 * n_fac) check
        # n=100, so 100*99/2 = 4950 pairs, each with 2 facility checks -> manageable
        
        best_swap_delta = 0.0
        best_swap_i = -1
        best_swap_j = -1
        
        # Get current assignment costs
        ac_i = assign_cost[np.arange(n_cust), fac_of]
        
        # For efficiency, iterate over pairs but skip if same facility
        # Group customers by facility
        fac_groups = {}
        for i in range(n_cust):
            f = fac_of[i]
            if f not in fac_groups:
                fac_groups[f] = []
            fac_groups[f].append(i)
        
        # For each pair of facilities, check cross-pairs
        facs_list = list(fac_groups.keys())
        for fi_idx in range(len(facs_list)):
            if time.time() > deadline - 3:
                break
            for fj_idx in range(fi_idx + 1, len(facs_list)):
                if time.time() > deadline - 3:
                    break
                fi = facs_list[fi_idx]
                fj = facs_list[fj_idx]
                group_i = fac_groups[fi]
                group_j = fac_groups[fj]
                
                # For each i in group_i, j in group_j:
                # delta = ac[j,fi] + ac[i,fj] - ac[i,fi] - ac[j,fj]
                # Capacity: after swap, load[fi] changes by -demands[i]+demands[j]
                #           load[fj] changes by -demands[j]+demands[i]
                # Need: load[fi] - demands[i] + demands[j] <= capacities[fi]
                #       load[fj] - demands[j] + demands[i] <= capacities[fj]
                
                for i in group_i:
                    if time.time() > deadline - 3:
                        break
                    for j in group_j:
                        # Capacity check
                        if load[fi] - demands[i] + demands[j] > capacities[fi]:
                            continue
                        if load[fj] - demands[j] + demands[i] > capacities[fj]:
                            continue
                        
                        delta = assign_cost[j, fi] + assign_cost[i, fj] - assign_cost[i, fi] - assign_cost[j, fj]
                        
                        if delta < best_swap_delta - 1e-9:
                            best_swap_delta = delta
                            best_swap_i = i
                            best_swap_j = j
                
                if best_swap_delta < -1e-9:
                    # Apply best swap found so far in this facility pair
                    # Actually, let's collect all and pick global best
                    pass
        
        # Apply best swap if found
        if best_swap_i != -1 and best_swap_j != -1:
            fi = fac_of[best_swap_i]
            fj = fac_of[best_swap_j]
            result[best_swap_i] = fj
            result[best_swap_j] = fi
            load[fi] -= demands[best_swap_i]
            load[fi] += demands[best_swap_j]
            load[fj] -= demands[best_swap_j]
            load[fj] += demands[best_swap_i]
        
        # If no improvement in either phase, we can stop early
        # (already handled by the single-move phase breaking out)
    
    return result

# EVOLVE-BLOCK-END
