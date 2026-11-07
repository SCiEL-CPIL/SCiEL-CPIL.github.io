"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def compute_cost(result):
        """Compute total cost for an assignment."""
        cost = 0.0
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            f = int(result[i])
            cost += assign_cost[i, f]
            load[f] += demands[i]
        for f in range(n_facilities):
            if load[f] > 1e-9:
                cost += open_cost[f]
        return cost, load
    
    def solve_lp_relaxation():
        """
        Solve LP relaxation of CFLP.
        Variables: x[i,f] = fraction of customer i assigned to facility f
                  y[f] = 1 if facility f is open
        Minimize: sum(i,f) assign_cost[i,f]*x[i,f] + sum(f) open_cost[f]*y[f]
        Subject to:
                  sum(f) x[i,f] = 1 for all i
                  sum(i) demands[i]*x[i,f] <= capacities[f]*y[f] for all f
                  0 <= x[i,f] <= 1
                  0 <= y[f] <= 1
        """
        # Variables: x[0..n_customers*n_facilities-1], y[n_customers*n_facilities .. n_customers*n_facilities+n_facilities-1]
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective
        c = np.zeros(n_vars)
        for i in range(n_customers):
            for f in range(n_facilities):
                c[i * n_facilities + f] = assign_cost[i, f]
        for f in range(n_facilities):
            c[n_x + f] = open_cost[f]
        
        # Equality constraints: sum_f x[i,f] = 1 for each customer i
        A_eq = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            for f in range(n_facilities):
                A_eq[i, i * n_facilities + f] = 1.0
        b_eq = np.ones(n_customers)
        
        # Inequality constraints: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
        A_ub = np.zeros((n_facilities, n_vars))
        for f in range(n_facilities):
            for i in range(n_customers):
                A_ub[f, i * n_facilities + f] = demands[i]
            A_ub[f, n_x + f] = -capacities[f]
        b_ub = np.zeros(n_facilities)
        
        # Bounds
        bounds = []
        for i in range(n_customers):
            for f in range(n_facilities):
                bounds.append((0, 1))
        for f in range(n_facilities):
            bounds.append((0, 1))
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        
        if not result.success:
            return None, None
        
        x = result.x[:n_x].reshape(n_customers, n_facilities)
        y = result.x[n_x:n_x + n_y]
        return x, y
    
    def round_lp_solution(x, y, time_limit):
        """
        Deterministic rounding of LP solution respecting capacity constraints.
        Uses a greedy approach guided by LP fractional values.
        """
        n_customers_local = n_customers
        n_facilities_local = n_facilities
        
        # Sort customers by how "difficult" they are (higher max fractional value = more certain)
        # Actually, sort by minimum fractional value - customers with low max x are more uncertain
        max_frac = np.max(x, axis=1)
        # Customers with lower max fractional value are more spread out and harder to assign
        # We want to assign the "most determined" customers first
        order = np.argsort(-max_frac)
        
        result = np.full(n_customers_local, -1, dtype=int)
        load = np.zeros(n_facilities_local)
        
        # First pass: assign customers with high confidence (max_frac > 0.8)
        for i in order:
            if time.time() > time_limit:
                break
            if result[i] != -1:
                continue
            if max_frac[i] > 0.8:
                # Assign to the facility with highest fractional value
                best_f = int(np.argmax(x[i]))
                if load[best_f] + demands[i] <= capacities[best_f] + 1e-6:
                    result[i] = best_f
                    load[best_f] += demands[i]
        
        # Second pass: assign remaining customers using LP-guided greedy
        for i in order:
            if time.time() > time_limit:
                break
            if result[i] != -1:
                continue
            
            # Find best feasible facility using LP values as guide
            best_f = -1
            best_score = np.inf
            for f in range(n_facilities_local):
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    # Score: LP value (higher is better) + small penalty for capacity usage
                    score = -x[i, f] + 0.001 * (load[f] + demands[i]) / (capacities[f] + 1e-9)
                    if score < best_score:
                        best_score = score
                        best_f = f
            
            if best_f == -1:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities - load
                best_f = int(np.argmax(remaining))
                if load[best_f] + demands[i] > capacities[best_f] + 1e-6:
                    # Last resort
                    best_f = 0
            
            result[i] = best_f
            load[best_f] += demands[i]
        
        # Verify all assigned
        if np.any(result == -1):
            # Force assign any unassigned
            for i in range(n_customers_local):
                if result[i] == -1:
                    best_f = 0
                    for f in range(n_facilities_local):
                        if load[f] + demands[i] <= capacities[f] + 1e-6:
                            best_f = f
                            break
                    result[i] = best_f
                    load[best_f] += demands[i]
        
        return result.astype(int), load
    
    def local_search(result, load, time_limit):
        """Local search with 1-move improvements, vectorized where possible."""
        current_cost, _ = compute_cost(result)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_customers):
                if time.time() > time_limit:
                    break
                old_f = result[i]
                best_delta = 0
                best_new_f = -1
                
                old_load = load[old_f]
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                        continue
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    if old_load == demands[i]:
                        delta_open -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                    
                    delta_cost = delta_assign + delta_open
                    if delta_cost < best_delta - 1e-9:
                        best_delta = delta_cost
                        best_new_f = new_f
                
                if best_new_f != -1:
                    load[old_f] -= demands[i]
                    load[best_new_f] += demands[i]
                    result[i] = best_new_f
                    current_cost += best_delta
                    improved = True
        
        return result, current_cost, load
    
    def swap_search(result, load, current_cost, time_limit):
        """Local search with 2-swap improvements."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_customers):
                if time.time() > time_limit:
                    break
                old_f_i = result[i]
                for j in range(i + 1, n_customers):
                    if time.time() > time_limit:
                        break
                    old_f_j = result[j]
                    if old_f_i == old_f_j:
                        continue
                    
                    if load[old_f_j] - demands[j] + demands[i] <= capacities[old_f_j] + 1e-6 and \
                       load[old_f_i] - demands[i] + demands[j] <= capacities[old_f_i] + 1e-6:
                        
                        delta_assign = (assign_cost[i, old_f_j] - assign_cost[i, old_f_i]) + \
                                       (assign_cost[j, old_f_i] - assign_cost[j, old_f_j])
                        
                        delta_open = 0.0
                        new_load_i = load[old_f_i] - demands[i] + demands[j]
                        new_load_j = load[old_f_j] - demands[j] + demands[i]
                        
                        if new_load_i == 0 and load[old_f_i] != 0:
                            delta_open -= open_cost[old_f_i]
                        if new_load_j == 0 and load[old_f_j] != 0:
                            delta_open -= open_cost[old_f_j]
                        if new_load_i > 0 and load[old_f_i] == 0:
                            delta_open += open_cost[old_f_i]
                        if new_load_j > 0 and load[old_f_j] == 0:
                            delta_open += open_cost[old_f_j]
                        
                        delta_cost = delta_assign + delta_open
                        if delta_cost < -1e-9:
                            load[old_f_i] -= demands[i]
                            load[old_f_j] -= demands[j]
                            load[old_f_j] += demands[i]
                            load[old_f_i] += demands[j]
                            result[i] = old_f_j
                            result[j] = old_f_i
                            current_cost += delta_cost
                            improved = True
                            break
                if time.time() > time_limit:
                    break
        
        return result, current_cost, load
    
    def greedy_fallback():
        """Simple greedy assignment as fallback."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    cost = assign_cost[i, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                remaining = capacities - load
                best_f = int(np.argmax(remaining))
            result[i] = best_f
            load[best_f] += demands[i]
        return result, load
    
    # Main algorithm
    best_result = None
    best_cost = np.inf
    best_load = None
    
    # Try LP relaxation approach
    x_lp, y_lp = solve_lp_relaxation()
    
    if x_lp is not None:
        # Round LP solution
        result, load = round_lp_solution(x_lp, y_lp, deadline - 60)
        cost, load = compute_cost(result)
        
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
        
        # Local search polish
        time_limit_ls = deadline - 40
        result, cost, load = local_search(result, load, time_limit_ls)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
        
        # Swap search
        time_limit_swap = deadline - 20
        result, cost, load = swap_search(result, load, best_cost, time_limit_swap)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
        
        # Final polish
        time_limit_final = deadline - 5
        result, cost, load = local_search(result, load, time_limit_final)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
    
    # If LP failed or result is bad, try greedy
    if best_result is None or best_cost > np.inf * 0.9:
        result, load = greedy_fallback()
        cost, load = compute_cost(result)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
        
        # Local search on greedy
        time_limit_ls = deadline - 30
        result, cost, load = local_search(result, load, time_limit_ls)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
        
        # Swap search
        time_limit_swap = deadline - 10
        result, cost, load = swap_search(result, load, best_cost, time_limit_swap)
        if cost < best_cost:
            best_result = result.copy()
            best_cost = cost
            best_load = load.copy()
    
    # Verify result
    if best_result is None:
        best_result, best_load = greedy_fallback()
    
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(best_result[i])
        if f < 0 or f >= n_facilities:
            best_result, best_load = greedy_fallback()
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[int(best_result[i])] += demands[i]
            break
        load_check[f] += demands[i]
    
    if np.any(load_check > capacities + 1e-6):
        best_result, best_load = greedy_fallback()
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
