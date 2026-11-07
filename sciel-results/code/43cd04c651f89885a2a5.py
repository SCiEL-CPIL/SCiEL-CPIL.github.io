import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # --- Helper: Compute cost of a solution ---
    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    # --- Helper: Greedy Initialization ---
    def greedy_init(order_func, randomize=False):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(order_func(demands))
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                perturbed = assign_cost[i] + np.random.uniform(0, 1) * 10
                fac_order = np.argsort(perturbed)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: find any feasible facility
                remaining = capacities - load
                feasible = np.where(remaining >= demands[i])[0]
                if len(feasible) > 0:
                    # Pick min cost among feasible
                    best_f = feasible[np.argmin(assign_cost[i, feasible])]
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    return None
            
        return result

    # --- Helper: LP Rounding Initialization ---
    def lp_rounding_init(lp_sol):
        if lp_sol is None:
            return None
        x_lp = lp_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            # Get facilities sorted by LP value descending
            sorted_facs = np.argsort(-x_lp[i])
            assigned = False
            for f in sorted_facs:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback to min cost feasible
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[f_best] = f_best # Wait, result[i]
                    result[i] = f_best
                else:
                    return None
        
        return result

    # --- 1. Generate Initial Candidates ---
    candidates = []
    
    # Strategy A: LP Relaxation
    lp_solution = None
    try:
        if time.time() < deadline - 20:
            # Variables: x[i, j] (n_cust * n_fac) and y[j] (n_fac)
            c_obj = np.concatenate([assign_cost.flatten(), open_cost])
            bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
            
            A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
            for i in range(n_cust):
                for j in range(n_fac):
                    A_eq[i, i * n_fac + j] = 1.0
            b_eq = np.ones(n_cust)
            
            n_cons_ub = n_fac + n_cust * n_fac
            A_ub = np.zeros((n_cons_ub, n_cust * n_fac + n_fac))
            b_ub = np.zeros(n_cons_ub)
            
            for j in range(n_fac):
                row_idx = j
                for i in range(n_cust):
                    A_ub[row_idx, i * n_fac + j] = demands[i]
                A_ub[row_idx, n_cust * n_fac + j] = -capacities[j]
                
            for i in range(n_cust):
                for j in range(n_fac):
                    row_idx = n_fac + i * n_fac + j
                    A_ub[row_idx, i * n_fac + j] = 1.0
                    A_ub[row_idx, n_cust * n_fac + j] = -1.0
                    
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                lp_solution = res.x
    except Exception:
        lp_solution = None
        
    init_lp = lp_rounding_init(lp_solution)
    if init_lp is not None:
        candidates.append(init_lp)
        
    # Strategy B: Greedy (Largest Demand First)
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)
        
    # Strategy C: Greedy (Smallest Demand First)
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)
        
    # Strategy D: Randomized Greedy
    for _ in range(3):
        if time.time() > deadline - 25:
            break
        init_r = greedy_init(lambda d: -d, randomize=True)
        if init_r is not None:
            candidates.append(init_r)
            
    if not candidates:
        # Last resort
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)

    # Pick best initial candidate
    best_init = min(candidates, key=compute_cost)
    
    # --- 2. Local Search Polish ---
    def _local_search(result, time_limit):
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        # Precompute arange for speed
        idx = np.arange(n_cust)
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (Move)
            # Iterate customers in random order to avoid bias? Or standard order?
            # Standard order is fine, but maybe shuffle occasionally?
            # Let's stick to standard for determinism in local search steps, 
            # but the multiple starts provide diversity.
            
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                    
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                # Vectorized delta calculation for speed
                # delta = assign_cost[i, :] - assign_cost[i, cur_f]
                # + open_cost changes
                # open_cost change:
                # if fac_count[cur_f] == 1: -open_cost[cur_f]
                # if fac_count[new_f] == 0: +open_cost[new_f]
                
                base_delta = assign_cost[i, :] - assign_cost[i, cur_f]
                
                # Mask infeasible facilities
                feasible_mask = (load + d_i) <= capacities
                feasible_mask[cur_f] = False # Don't move to same
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate open cost adjustment
                # This is tricky to vectorize cleanly because it depends on fac_count[new_f]==0
                # and fac_count[cur_f]==1.
                
                # Let's do a semi-vectorized approach
                # Start with base assignment cost difference
                deltas = base_delta.copy()
                
                # If moving from a facility that becomes empty, we save its open cost
                if fac_count[cur_f] == 1:
                    deltas -= open_cost[cur_f]
                    
                # If moving to a facility that is currently empty, we pay its open cost
                empty_facs = np.where(fac_count == 0)[0]
                if len(empty_facs) > 0:
                    deltas[empty_facs] += open_cost[empty_facs]
                    
                # Apply feasibility mask
                deltas[~feasible_mask] = np.inf
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < -1e-9:
                    best_new_f = min_idx
                    best_delta = min_delta
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                        
            # 2. Swap
            # To speed up, we can just iterate all pairs. n=100, pairs=4950.
            # We can vectorize this too?
            # For each pair (i, j), delta = (cost[i, f_j] + cost[j, f_i]) - (cost[i, f_i] + cost[j, f_j])
            # Capacity checks:
            # load[f_i] - d_i + d_j <= cap[f_i]
            # load[f_j] - d_j + d_i <= cap[f_j]
            
            # Let's do a simple loop but break early if time is tight.
            # We can optimize by only checking pairs where f_i != f_j.
            
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change for swap between different facilities is 0
                    # (counts stay same)
                    
                    if delta < -1e-9:
                        # Apply swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        # Continue searching for more swaps in this pass
                        # Don't break, just update state and continue to next j
                        # Note: updating load/current inside the loop is safe as long as we
                        # re-fetch f_i, f_j, d_i, d_j for subsequent iterations?
                        # Actually, if we swap i and j, their assignments change.
                        # The inner loop continues with j+1.
                        # The current[i] is now f_j.
                        # The current[j] is now f_i.
                        # This is fine.
                        
        return best_result

    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Run local search with a generous time limit
        final_result = _local_search(best_init, remaining_time - 5)
    else:
        final_result = best_init
        
    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return best_init # Fallback to valid initial
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        return best_init
        
    return final_result
