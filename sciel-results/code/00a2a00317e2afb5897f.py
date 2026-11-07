import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # --- 1. LP Relaxation for Initialization ---
    # Variables: x[i, j] (n_cust * n_fac) and y[j] (n_fac)
    # Total variables: n_cust * n_fac + n_fac
    # Constraints:
    # 1. Sum_j x[i, j] == 1 for all i (n_cust constraints)
    # 2. Sum_i d[i] * x[i, j] - C[j] * y[j] <= 0 for all j (n_fac constraints)
    # 3. x[i, j] - y[j] <= 0 for all i, j (n_cust * n_fac constraints)
    
    # We can reduce constraints 3 by noting that if x[i,j] > 0 then y[j] must be 1.
    # In LP relaxation, y[j] can be fractional.
    # Let's set up the matrix A_ub and b_ub.
    
    # Objective: min sum c[i,j] x[i,j] + sum f[j] y[j]
    c_obj = np.concatenate([assign_cost.flatten(), open_cost])
    
    # Bounds: 0 <= x[i,j] <= 1, 0 <= y[j] <= 1
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
    
    # A_eq: Sum_j x[i,j] = 1
    # Size: n_cust x (n_cust*n_fac + n_fac)
    A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    b_eq = np.ones(n_cust)
    
    # A_ub:
    # Part 1: Capacity constraints: Sum_i d[i] x[i,j] - C[j] y[j] <= 0
    # Part 2: Linking constraints: x[i,j] - y[j] <= 0
    
    n_cons_ub = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_cons_ub, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_cons_ub)
    
    # Fill capacity constraints
    for j in range(n_fac):
        row_idx = j
        for i in range(n_cust):
            A_ub[row_idx, i * n_fac + j] = demands[i]
        A_ub[row_idx, n_cust * n_fac + j] = -capacities[j]
        # b_ub[row_idx] is 0
    
    # Fill linking constraints
    for i in range(n_cust):
        for j in range(n_fac):
            row_idx = n_fac + i * n_fac + j
            A_ub[row_idx, i * n_fac + j] = 1.0
            A_ub[row_idx, n_cust * n_fac + j] = -1.0
            # b_ub[row_idx] is 0
            
    # Solve LP
    lp_solution = None
    try:
        if time.time() < deadline - 10: # Leave time for repair and search
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                lp_solution = res.x
    except Exception:
        lp_solution = None
        
    # --- 2. Initialization from LP or Greedy ---
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    if lp_solution is not None:
        x_lp = lp_solution[:n_cust * n_fac].reshape(n_cust, n_fac)
        # Sort customers by demand descending to prioritize large ones
        order = np.argsort(-demands)
        
        for i in order:
            # Get facilities sorted by LP value descending
            # Tie-break by assign_cost
            # Create a combined score: -x_lp[i] * large_factor + assign_cost[i]
            # Or simply sort by x_lp[i] descending, then by assign_cost[i] ascending
            fac_indices = np.arange(n_fac)
            # Primary key: -x_lp[i, fac_indices]
            # Secondary key: assign_cost[i, fac_indices]
            # We want min cost among high LP values.
            # Let's just sort by LP value descending.
            sorted_facs = fac_indices[np.argsort(-x_lp[i])]
            
            assigned = False
            for f in sorted_facs:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: find any feasible facility with min cost
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[i] = f_best
                else:
                    # Should not happen if instance is feasible
                    result[i] = 0
    else:
        # Fallback: Greedy Largest Demand First, Min Cost
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                # Fallback
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    result[i] = 0

    # --- 3. Local Search Polish ---
    # Fast local search with delta cost updates.
    # Moves: 1-move (reassign) and 2-swap.
    
    def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
        n_cust = len(demands)
        n_fac = len(capacities)
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
        
        # Pre-compute sorted facility indices by cost for each customer for faster 1-move checks?
        # For 1-move, we check all other facilities. n_fac=20, so it's cheap.
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    # Open cost change
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
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
            # Try swapping pairs (i, j) where current[i] != current[j]
            # To speed up, we can just iterate all pairs. n=100, pairs=4950.
            # For each pair, check capacity feasibility and cost delta.
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
                    # f_i loses d_i, gains d_j
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    # f_j loses d_j, gains d_i
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change for swap between different facilities
                    # fac_count[f_i] and fac_count[f_j] remain unchanged (lose 1, gain 1)
                    # So delta open cost is 0
                    
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
                        break # Restart j loop for this i? Or just continue? 
                        # Breaking j loop for this i is safer to avoid conflicting updates in same pass
                        # But we are doing best-improvement in a pass? No, we are doing first-improvement within the pass?
                        # Actually, the loop structure above applies the FIRST improvement found for i in 1-move.
                        # For swaps, it applies the FIRST improvement found for pair (i,j).
                        # This is "First Improvement" local search.
                        
        return best_result

    remaining_time = deadline - time.time()
    if remaining_time > 5:
        final_result = _local_search(result, demands, capacities, open_cost, assign_cost, remaining_time - 2)
    else:
        final_result = result
        
    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int) # Invalid fallback
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        # If invalid, return the initial result which was valid
        return result
        
    return final_result
