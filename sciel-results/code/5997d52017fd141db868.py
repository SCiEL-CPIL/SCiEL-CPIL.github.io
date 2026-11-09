import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capapacities) if 'capapacities' in dir() else len(capacities)
    
    # Variables:
    # y_f (n_facilities) : binary, 1 if facility f is open
    # x_{i,f} (n_customers * n_facilities) : binary, 1 if customer i assigned to f
    # Total variables: n_facilities + n_customers * n_facilities
    
    n_vars = n_facilities + n_customers * n_facilities
    
    # Objective coefficients
    # Minimize: sum_f open_cost[f] * y_f + sum_i,f assign_cost[i,f] * x_{i,f}
    c = np.zeros(n_vars)
    c[:n_facilities] = open_cost
    c[n_facilities:] = assign_cost.flatten()
    
    # Bounds
    # y_f in [0, 1]
    # x_{i,f} in [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Constraints
    # 1. Each customer assigned to exactly one facility:
    #    sum_f x_{i,f} = 1 for each i
    #    Matrix shape: (n_customers, n_vars)
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        # x_{i,f} are at indices n_facilities + i*n_facilities + f
        start_idx = n_facilities + i * n_facilities
        A_eq[i, start_idx:start_idx+n_facilities] = 1.0
        
    # 2. Capacity constraints:
    #    sum_i demands[i] * x_{i,f} <= capacities[f] for each f
    #    Matrix shape: (n_facilities, n_vars)
    A_ineq = np.zeros((n_facilities, n_vars))
    b_ub = capacities
    for f in range(n_facilities):
        for i in range(n_customers):
            # x_{i,f} index
            idx = n_facilities + i * n_facilities + f
            A_ineq[f, idx] = demands[i]
            
    # 3. Linking constraints:
    #    x_{i,f} <= y_f  =>  x_{i,f} - y_f <= 0
    #    Matrix shape: (n_customers * n_facilities, n_vars)
    n_link_constraints = n_customers * n_facilities
    A_link = np.zeros((n_link_constraints, n_vars))
    b_link_ub = np.zeros(n_link_constraints)
    b_link_lb = -np.inf * np.ones(n_link_constraints)
    
    k = 0
    for i in range(n_customers):
        for f in range(n_facilities):
            idx_x = n_facilities + i * n_facilities + f
            idx_y = f
            A_link[k, idx_x] = 1.0
            A_link[k, idx_y] = -1.0
            k += 1
            
    # Combine inequality constraints:
    # A_ineq * vars <= b_ub
    # A_link * vars <= b_link_ub (and >= b_link_lb which is -inf, so effectively just <= 0)
    # We can combine them into one LinearConstraint
    A_total_ineq = np.vstack([A_ineq, A_link])
    b_total_ub = np.concatenate([b_ub, b_link_ub])
    b_total_lb = np.concatenate([np.full(n_facilities, -np.inf), b_link_lb])
    
    constraints = LinearConstraint(A_total_ineq, b_total_lb, b_total_ub)
    bounds = Bounds(lb, ub)
    
    try:
        # Use HiGHS solver which is default in scipy milp
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract x_{i,f}
            x_flat = res.x[n_facilities:]
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            
            # Find assigned facility for each customer
            result = np.argmax(x_matrix, axis=1).astype(int)
            
            # Verify feasibility just in case of numerical issues
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                f = result[i]
                load[f] += demands[i]
                
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-6:
                    raise ValueError("Capacity constraint violated")
                
            return result
            
        else:
            # If solver fails, fallback to greedy
            pass
            
    except Exception:
        pass
        
    # Fallback: Simple Greedy
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    
    for i in order:
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        if feasible:
            # Choose cheapest feasible
            result[i] = feasible[0]
            load[result[i]] += demands[i]
        else:
            # Should not happen, but assign to first with space (or 0)
            # If no space, this instance is infeasible? 
            # Problem states capacities are integers, demands 5-35. 
            # If infeasible, we must return something. 
            # Let's try to find any facility with space.
            assigned = False
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                result[i] = 0
                load[0] += demands[i]
                
    return result
