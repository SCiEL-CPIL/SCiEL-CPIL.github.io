import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initial assignment: try to assign each customer to cheapest feasible facility
    # Use a more sophisticated initialization
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand (largest first) for better packing
    order = np.argsort(-demands)
    
    for i in order:
        # Try facilities in order of assignment cost
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        if feasible:
            # Choose the one with best "effective cost" considering open cost
            best_f = feasible[0]
            best_cost = assign_cost[i, best_f]
            for f in feasible[1:]:
                # If facility is new, add open cost
                extra_open = open_cost[f] if load[f] == 0 else 0
                cost = assign_cost[i, f] + extra_open
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
            result[i] = best_f
            load[best_f] += demands[i]
        else:
            # Should not happen if feasible, but fallback
            result[i] = 0
            load[0] += demands[i]
    
    # Local search improvement
    def total_cost(assignment):
        cost = 0.0
        for f in range(n_facilities):
            customers_at_f = np.where(assignment == f)[0]
            if len(customers_at_f) > 0:
                cost += open_cost[f]
                cost += np.sum(assign_cost[customers_at_f, f])
        return cost
    
    current_cost = total_cost(result)
    improved = True
    max_iter = 50
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        # Try moving each customer to a different facility
        for i in range(n_customers):
            old_f = result[i]
            if old_f < 0:
                continue
            for new_f in range(n_facilities):
                if new_f == old_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                # Calculate cost change
                old_cost = assign_cost[i, old_f]
                new_cost = assign_cost[i, new_f]
                
                # Open cost changes
                old_open = open_cost[old_f] if (load[old_f] - demands[i]) > 0 else 0
                new_open = open_cost[new_f] if load[new_f] > 0 else 0
                
                # If old facility becomes empty, we save its open cost
                old_total = old_cost + (open_cost[old_f] if (load[old_f] - demands[i]) == 0 else 0)
                new_total = new_cost + (open_cost[new_f] if load[new_f] == 0 else 0)
                
                delta = new_total - old_total
                if delta < -1e-9:
                    # Apply move
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    current_cost += delta
                    improved = True
                    break
        
        # Try swapping two customers at different facilities
        for i in range(n_customers):
            for j in range(i+1, n_customers):
                if result[i] == result[j] or result[i] < 0 or result[j] < 0:
                    continue
                f_i = result[i]
                f_j = result[j]
                
                # Check if swap is feasible
                if f_i == f_j:
                    continue
                # After swap: i goes to f_j, j goes to f_i
                # Capacity check: f_j must have room for demands[i] after removing demands[j]
                # f_i must have room for demands[j] after removing demands[i]
                if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                    continue
                if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                    continue
                
                # Cost change
                old_cost = assign_cost[i, f_i] + assign_cost[j, f_j]
                new_cost = assign_cost[i, f_j] + assign_cost[j, f_i]
                
                # Open cost: both facilities still have customers (since we're swapping, both remain non-empty)
                # So open cost doesn't change
                delta = new_cost - old_cost
                if delta < -1e-9:
                    # Apply swap
                    load[f_i] -= demands[i]
                    load[f_j] -= demands[j]
                    load[f_j] += demands[i]
                    load[f_i] += demands[j]
                    result[i] = f_j
                    result[j] = f_i
                    current_cost += delta
                    improved = True
    
    # Verify feasibility
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return np.zeros(n_customers, dtype=int)
        load_check[f] += demands[i]
    
    for f in range(n_facilities):
        if load_check[f] > capacities[f]:
            # Fallback to simple greedy
            result = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities)
            for i in np.argsort(-demands):
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            return result
    
    return result
