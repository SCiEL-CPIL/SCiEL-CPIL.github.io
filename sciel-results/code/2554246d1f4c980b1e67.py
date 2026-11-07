"""Solve capacitated facility location via MILP (HiGHS) with greedy fallback and local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _greedy_fallback(demands, capacities, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            result[i] = 0
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, max_iters=500, time_limit=10.0):
    """Local search: try moving customers to reduce cost while respecting capacity."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    improved = True
    iter_count = 0
    start_time = time.time()
    
    while improved and iter_count < max_iters and (time.time() - start_time) < time_limit:
        improved = False
        iter_count += 1
        # Try moving each customer to a different facility
        for i in range(n_customers):
            current_f = result[i]
            # Try all other facilities
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                # Calculate cost change
                # Remove customer i from current_f
                old_load = load[current_f]
                new_load_current = old_load - demands[i]
                new_load_new = load[new_f] + demands[i]
                
                # Change in assignment cost
                delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                # Change in opening cost
                delta_open = 0.0
                if new_load_current == 0:
                    # Facility current_f closes
                    delta_open -= open_cost[current_f]
                if new_load_new > 0 and load[new_f] == 0:
                    # Facility new_f opens
                    delta_open += open_cost[new_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Apply move
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    improved = True
                    break
        # Early stop if no improvement in this pass? No, continue to next customer.
        # Actually, the loop above breaks on first improvement for customer i, then moves to i+1.
        # This is a steepest descent variant.
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Try MILP first
    try:
        # Variables:
        # x[i, f] = 1 if customer i assigned to facility f (n_customers * n_facilities)
        # y[f] = 1 if facility f is open (n_facilities)
        # Total variables: n_customers * n_facilities + n_facilities
        
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective: minimize sum(open_cost[f] * y[f]) + sum(assign_cost[i,f] * x[i,f])
        c = np.zeros(n_vars)
        for i in range(n_customers):
            for f in range(n_facilities):
                c[i * n_facilities + f] = assign_cost[i, f]
        for f in range(n_facilities):
            c[n_customers * n_facilities + f] = open_cost[f]
        
        # Integrality: x and y are binary
        integrality = np.ones(n_vars)
        
        # Constraints:
        # 1. Each customer assigned to exactly one facility:
        #    sum_f x[i,f] = 1 for all i
        # 2. Capacity: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
        #    => sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
        
        n_constraints = n_customers + n_facilities
        
        # Build constraint matrix A for: lb <= A @ var <= ub
        # We'll use LinearConstraint
        A = np.zeros((n_constraints, n_vars))
        lb = np.zeros(n_constraints)
        ub = np.zeros(n_constraints)
        
        # Constraint 1: sum_f x[i,f] = 1
        for i in range(n_customers):
            for f in range(n_facilities):
                A[i, i * n_facilities + f] = 1.0
            lb[i] = 1.0
            ub[i] = 1.0
        
        # Constraint 2: sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
        for f in range(n_facilities):
            row = n_customers + f
            for i in range(n_customers):
                A[row, i * n_facilities + f] = demands[i]
            A[row, n_customers * n_facilities + f] = -capacities[f]
            lb[row] = -np.inf
            ub[row] = 0.0
        
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            x = res.x
            # Extract assignment
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                best_f = np.argmax(x[i * n_facilities:(i+1) * n_facilities])
                result[i] = best_f
            
            # Validate
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            if np.all(load <= capacities + 1e-6):
                # Apply local search to improve
                result = _local_search(demands, capacities, open_cost, assign_cost, result)
                return result
            else:
                # Capacity violated, fall through to greedy
                pass
    except Exception:
        pass
    
    # Check if we're still within deadline
    if time.time() > deadline:
        return _greedy_fallback(demands, capacities, assign_cost)
    
    # Fallback to greedy
    result = _greedy_fallback(demands, capacities, assign_cost)
    # Apply local search to greedy result
    result = _local_search(demands, capacities, open_cost, assign_cost, result)
    return result
# EVOLVE-BLOCK-END
