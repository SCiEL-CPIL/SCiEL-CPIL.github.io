import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    def calc_cost(assignment, load):
        total = 0.0
        for j in range(n_facilities):
            if load[j] > 0:
                total += open_cost[j]
        for i in range(n_customers):
            j = assignment[i]
            if j >= 0:
                total += assign_cost[i, j]
        return total

    def valid_assignment(assignment, load):
        if np.any(load < -1e-9):
            return False
        if np.any(load > capacities + 1e-9):
            return False
        if np.any(assignment < 0) or np.any(assignment >= n_facilities):
            return False
        return True

    # Strategy: Solve a min-cost flow formulation where each customer sends
    # 'demand[i]' units to a facility. The fixed opening cost is approximated
    # by a per-unit cost on the first unit sent to a facility, or more simply,
    # we solve an LP that includes the fixed charge via a big-M or similar, 
    # but since we can't do integer programming easily, we use the LP relaxation
    # of the CFLP directly as the primary solver, which is more powerful than 
    # greedy init.
    
    # We will solve the LP relaxation of the CFLP:
    # Min sum c_ij x_ij + sum f_j y_j
    # s.t. sum_j x_ij = 1
    #      sum_i d_i x_ij <= u_j y_j
    #      0 <= x_ij <= 1
    #      y_j in {0,1} -> relaxed to [0,1]
    
    # Note: The constraint sum_i d_i x_ij <= u_j y_j is non-linear in standard LP form if y_j is a variable.
    # Standard LP formulation for CFLP uses:
    # sum_i d_i x_ij <= u_j y_j is not linear if we treat it as product.
    # Wait, the constraint is:
    # sum_i d_i x_ij <= u_j * y_j
    # This is not linear because of y_j * u_j? No, u_j is constant.
    # But x_ij and y_j are variables. The constraint is linear: 
    # sum_i d_i x_ij - u_j y_j <= 0.
    # Yes, this is linear.
    
    # So we can solve the LP relaxation directly.
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective
    c = np.zeros(n_vars)
    c[:n_customers * n_facilities] = assign_cost.flatten()
    c[n_customers * n_facilities:] = open_cost
    
    # Constraints
    # 1. sum_j x_ij = 1 for each i
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    # 2. sum_i d_i x_ij - u_j y_j <= 0 for each j
    A_ub = np.zeros((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        
    bounds = [(0, 1)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            x_frac = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            y_frac = res.x[n_customers * n_facilities:]
            
            # Rounding strategy:
            # We need to convert fractional x to integer assignment.
            # A common heuristic is to sort customers by how "decisive" their assignment is
            # or by demand, and assign greedily respecting capacity, guided by LP values.
            
            # Let's try a greedy rounding guided by LP.
            # Sort facilities by y_frac descending (likely to be open)
            # Sort customers by demand descending (harder to place)
            
            # However, a better approach for CFLP rounding is often to use a 
            # "largest first" or "most fractional first" strategy.
            # Let's use a greedy method that respects capacity and uses LP values as tie-breakers/weights.
            
            assignment = np.full(n_customers, -1, dtype=int)
            load = np.zeros(n_facilities, dtype=np.float64)
            
            # Order customers: those with higher demand first, or those with more "spread" in LP?
            # Let's stick to demand descending for stability.
            cust_order = np.argsort(-demands)
            
            for i in cust_order:
                d_i = demands[i]
                best_j = -1
                best_score = float('inf')
                
                for j in range(n_facilities):
                    if load[j] + d_i <= capacities[j]:
                        # Score: LP value is high -> good. 
                        # We want to minimize cost, so we prefer high x_frac[i,j]
                        # But we also need to consider the actual cost.
                        # The LP objective already minimized cost.
                        # So we should prioritize assignments with high x_frac[i,j].
                        # If x_frac is close to 1, it's almost certainly the optimal choice.
                        
                        # Let's use a score that balances LP preference and actual cost.
                        # Actually, just picking the j with max x_frac[i,j] among feasible ones
                        # is a standard rounding heuristic (Largest First / LP-guided).
                        
                        # But what if x_frac[i,j] is 0.5 for two facilities?
                        # Tie-break with assign_cost.
                        
                        score = -x_frac[i, j]  # Minimize negative LP value (i.e., maximize LP value)
                        # Add small penalty for cost to break ties?
                        # Or just use x_frac. If x_frac is 0, we shouldn't pick it unless forced.
                        
                        # Let's try: score = assign_cost[i, j] - lambda * x_frac[i, j]
                        # Lambda should be large enough to prefer LP solution.
                        # But assign_cost is in range [0, ~1400] (100 * sqrt(2)).
                        # x_frac is in [0, 1].
                        # Let's try lambda = 1000.
                        
                        score = assign_cost[i, j] - 1000.0 * x_frac[i, j]
                        
                        if score < best_score:
                            best_score = score
                            best_j = j
                            
                if best_j == -1:
                    # No feasible facility? This shouldn't happen if total demand <= total capacity.
                    # If it does, we have a problem. Force assign to the one with most remaining capacity.
                    # This might lead to invalid solution, but better than crash.
                    best_j = np.argmax(capacities - load)
                    
                assignment[i] = best_j
                load[best_j] += d_i
                
            # Check validity
            if valid_assignment(assignment, load):
                current_cost = calc_cost(assignment, load)
                
                # Local search refinement
                no_improve = 0
                max_no_improve = 20
                
                while time.time() < deadline and no_improve < max_no_improve:
                    improved = False
                    
                    # 1-moves
                    for i in range(n_customers):
                        if time.time() > deadline:
                            break
                        j = assignment[i]
                        d_i = demands[i]
                        best_k = -1
                        best_delta = -1e-10
                        for k in range(n_facilities):
                            if k == j:
                                continue
                            if load[k] + d_i <= capacities[k]:
                                delta = assign_cost[i, k] - assign_cost[i, j]
                                if load[j] - d_i == 0:
                                    delta -= open_cost[j]
                                if load[k] == 0:
                                    delta += open_cost[k]
                                if delta < best_delta:
                                    best_delta = delta
                                    best_k = k
                        if best_k != -1:
                            load[j] -= d_i
                            load[best_k] += d_i
                            assignment[i] = best_k
                            current_cost += best_delta
                            improved = True
                            
                    if not improved:
                        # 2-swaps
                        found_swap = False
                        for i in range(n_customers):
                            if found_swap:
                                break
                            j = assignment[i]
                            d_i = demands[i]
                            best_l = -1
                            best_delta = -1e-10
                            for l in range(i + 1, n_customers):
                                k = assignment[l]
                                if j == k:
                                    continue
                                d_l = demands[l]
                                new_load_j = load[j] - d_i + d_l
                                new_load_k = load[k] - d_l + d_i
                                if new_load_j <= capacities[j] and new_load_k <= capacities[k]:
                                    delta_assign = (assign_cost[i, k] - assign_cost[i, j]) + (assign_cost[l, j] - assign_cost[l, k])
                                    delta_open = 0.0
                                    if new_load_j == 0:
                                        delta_open -= open_cost[j]
                                    if new_load_k == 0:
                                        delta_open -= open_cost[k]
                                    total_delta = delta_assign + delta_open
                                    if total_delta < best_delta:
                                        best_delta = total_delta
                                        best_l = l
                            if best_l != -1:
                                l = best_l
                                k = assignment[l]
                                d_l = demands[l]
                                load[j] -= d_i
                                load[k] -= d_l
                                load[j] += d_l
                                load[k] += d_i
                                assignment[i] = k
                                assignment[l] = j
                                current_cost += best_delta
                                improved = True
                                found_swap = True
                                
                        if found_swap:
                            no_improve = 0
                            continue
                            
                    if improved:
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                return assignment
                
    except Exception:
        pass

    # Fallback: Greedy
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=np.float64)
    cust_order = np.argsort(-demands)
    for i in cust_order:
        d_i = demands[i]
        best_j = -1
        best_cost = float('inf')
        for j in range(n_facilities):
            if load[j] + d_i <= capacities[j]:
                mc = assign_cost[i, j]
                if load[j] == 0:
                    mc += open_cost[j]
                if mc < best_cost:
                    best_cost = mc
                    best_j = j
        if best_j == -1:
            best_j = np.argmax(capacities - load)
        assignment[i] = best_j
        load[best_j] += d_i
        
    return assignment
