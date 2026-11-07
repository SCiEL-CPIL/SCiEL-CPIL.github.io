import numpy as np
from scipy.optimize import linprog
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return max(0, time_limit - elapsed() - safety_margin)
    
    def compute_cost(assignment):
        # assignment: array of facility indices
        if len(assignment) != n_cust:
            return np.inf
        # Check validity
        load = np.zeros(n_fac)
        for i in range(n_cust):
            f = assignment[i]
            if f < 0 or f >= n_fac:
                return np.inf
            load[f] += demands[i]
        for j in range(n_fac):
            if load[j] > capacities[j] + 1e-9:
                return np.inf
        
        total = 0.0
        open_mask = np.zeros(n_fac, dtype=bool)
        for i in range(n_cust):
            f = assignment[i]
            total += assign_cost[i, f]
            open_mask[f] = True
        total += np.sum(open_cost[open_mask])
        return total
    
    def greedy_assign():
        """Greedy assignment: largest demand first, cheapest feasible facility."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Find feasible facilities
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                # Fallback: pick facility with most remaining capacity
                rem = capacities - load
                f = np.argmax(rem)
                result[i] = f
                load[f] += demands[i]
            else:
                # Pick cheapest among feasible
                costs = assign_cost[i, feasible]
                best_idx = feasible[np.argmin(costs)]
                result[i] = best_idx
                load[best_idx] += demands[i]
        return result
    
    def local_search(assignment, max_iter=5000):
        """Local search: try moving each customer to a different facility."""
        assignment = assignment.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if remaining() < 2.0:
                break
            for i in range(n_cust):
                if remaining() < 2.0:
                    break
                old_f = assignment[i]
                # Try moving to each other facility
                best_f = old_f
                best_cost = current_cost
                for new_f in range(n_fac):
                    if new_f == old_f:
                        continue
                    # Check capacity
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                        continue
                    # Compute cost change
                    # Remove customer i from old_f
                    old_load = load[old_f]
                    new_load = load[new_f]
                    
                    # Change in assignment cost
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    
                    # Change in open cost
                    delta_open = 0.0
                    # old_f becomes empty?
                    if old_load == demands[i]:
                        # old_f was only serving customer i, so it closes
                        delta_open -= open_cost[old_f]
                    # new_f was empty?
                    if new_load == 0:
                        # new_f opens
                        delta_open += open_cost[new_f]
                    
                    delta_total = delta_assign + delta_open
                    if delta_total < best_cost - current_cost - 1e-9:
                        best_cost = current_cost + delta_total
                        best_f = new_f
                
                if best_f != old_f:
                    # Make the move
                    load[old_f] -= demands[i]
                    load[best_f] += demands[i]
                    assignment[i] = best_f
                    current_cost = best_cost
                    improved = True
        
        return assignment
    
    # Phase 1: Greedy initialization
    assignment = greedy_assign()
    best_assignment = assignment.copy()
    best_cost = compute_cost(assignment)
    
    if remaining() < 10.0:
        return best_assignment
    
    # Phase 2: LP relaxation via linprog (HiGHS)
    # Variables: x_ij for i in 0..n_cust-1, j in 0..n_fac-1
    # x_ij is continuous in [0, 1]
    # Constraints:
    #   sum_j x_ij = 1 for each i (n_cust constraints)
    #   sum_i d_i * x_ij <= C_j for each j (n_fac constraints)
    # Objective: min sum_{i,j} c_ij * x_ij
    # 
    # Note: This LP relaxation does NOT include open costs directly.
    # The open cost is a fixed charge, which makes this a non-convex problem.
    # However, we can use the LP solution as a starting point for local search.
    
    n_vars = n_cust * n_fac
    c_obj = assign_cost.flatten()
    
    # Equality constraints: sum_j x_ij = 1
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    
    # Inequality constraints: sum_i d_i * x_ij <= C_j
    A_ub = np.zeros((n_fac, n_vars))
    b_ub = capacities.copy()
    for i in range(n_cust):
        for j in range(n_fac):
            A_ub[j, i * n_fac + j] = demands[i]
    
    bounds = [(0, 1)] * n_vars
    
    try:
        if remaining() > 30.0:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.status == 0 and res.x is not None:
                x_flat = res.x.reshape(n_cust, n_fac)
                # Extract assignment from LP solution
                # For each customer, pick the facility with highest x_ij
                lp_assignment = np.zeros(n_cust, dtype=int)
                for i in range(n_cust):
                    lp_assignment[i] = np.argmax(x_flat[i])
                
                # Check if LP solution is feasible (integer)
                # If not, we still use it as a starting point
                lp_cost = compute_cost(lp_assignment)
                if lp_cost < best_cost:
                    best_cost = lp_cost
                    best_assignment = lp_assignment.copy()
                
                # Run local search from LP solution
                if remaining() > 15.0:
                    ls_assignment = local_search(lp_assignment, max_iter=2000)
                    ls_cost = compute_cost(ls_assignment)
                    if ls_cost < best_cost:
                        best_cost = ls_cost
                        best_assignment = ls_assignment.copy()
    except Exception:
        pass
    
    # Phase 3: Local search from greedy solution
    if remaining() > 15.0:
        ls_assignment = local_search(assignment, max_iter=3000)
        ls_cost = compute_cost(ls_assignment)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_assignment = ls_assignment.copy()
    
    # Phase 4: Additional local search iterations if time permits
    if remaining() > 10.0:
        for _ in range(3):
            if remaining() < 5.0:
                break
            ls_assignment = local_search(best_assignment, max_iter=1000)
            ls_cost = compute_cost(ls_assignment)
            if ls_cost < best_cost - 1e-9:
                best_cost = ls_cost
                best_assignment = ls_assignment.copy()
            else:
                break
    
    return best_assignment
