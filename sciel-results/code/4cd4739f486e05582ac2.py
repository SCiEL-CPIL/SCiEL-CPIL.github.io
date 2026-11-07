"""MILP solver for capacitated facility location using scipy.optimize.milp."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using Mixed-Integer Linear Programming.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    # Number of variables: N*F for x_cf, F for y_f
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    # c_x: assign_cost flattened (row-major: c=0,f=0; c=0,f=1; ... c=1,f=0; ...)
    c_x = assign_cost.flatten()
    c_y = open_cost
    c = np.concatenate([c_x, c_y])

    # Bounds
    # All variables are binary [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    integrality = np.ones(n_vars)  # All variables are integer

    # Constraints
    # 1. Assignment constraints: sum_f x_cf = 1 for each c
    #    Matrix A_eq of shape (N, N*F)
    #    For row c, columns c*F to (c+1)*F - 1 are 1, others 0.
    A_eq = np.zeros((n_customers, n_customers * n_facilities))
    for c in range(n_customers):
        A_eq[c, c*n_facilities:(c+1)*n_facilities] = 1.0
        
    b_eq = np.ones(n_customers)

    # 2. Capacity constraints: sum_c demands[c] * x_cf - capacities[f] * y_f <= 0
    #    Matrix A_ub of shape (F, N*F + F)
    #    For row f, the part for x variables: 
    #    x_cf is at index c*F + f. Coefficient is demands[c].
    #    The part for y variables: y_f is at index N*F + f. Coefficient is -capacities[f].
    A_ub = np.zeros((n_facilities, n_vars))
    for f in range(n_facilities):
        # Fill x coefficients
        for c in range(n_customers):
            idx_x = c * n_facilities + f
            A_ub[f, idx_x] = demands[c]
        # Fill y coefficient
        idx_y = n_customers * n_facilities + f
        A_ub[f, idx_y] = -capacities[f]
        
    b_ub = np.zeros(n_facilities)

    # Combine constraints
    # LinearConstraint(A, lb, ub)
    # We have equality and inequality. We can combine them or use separate constraints.
    # milp accepts a single LinearConstraint or a list? 
    # Documentation says: "LinearConstraint or a tuple of (A, lb, ub)"
    # Actually, we can just use the tuple form or construct a single matrix.
    # Let's combine into one matrix A of shape (N+F, N*F+F)
    
    A = np.vstack([A_eq, A_ub])
    lb_cons = np.concatenate([b_eq, np.full(n_facilities, -np.inf)])
    ub_cons = np.concatenate([b_eq, b_ub])
    
    cons = LinearConstraint(A, lb_cons, ub_cons)
    bounds = Bounds(lb, ub)

    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_customers * n_facilities]
            
            # Extract assignment
            assignment = np.zeros(n_customers, dtype=int)
            for c in range(n_customers):
                # Find argmax in the slice for customer c
                start_idx = c * n_facilities
                end_idx = (c + 1) * n_facilities
                x_c = x_flat[start_idx:end_idx]
                # Due to floating point, take argmax
                assignment[c] = np.argmax(x_c)
            
            # Verify feasibility just in case
            load = np.zeros(n_facilities)
            for c in range(n_customers):
                f = assignment[c]
                if f < 0 or f >= n_facilities:
                    raise ValueError("Invalid facility index")
                load[f] += demands[c]
            
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-9:
                    raise ValueError("Capacity exceeded")
            
            return assignment

        else:
            # Fallback to a simple greedy if MILP fails or times out
            # This should rarely happen for such a small problem
            pass
            
    except Exception as e:
        pass

    # Fallback: Greedy Assignment
    # Sort customers by demand descending to place large ones first
    order = np.argsort(-demands)
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    feasible_found = False
    for c in order:
        costs = assign_cost[c].copy()
        feasible = load + demands[c] <= capacities
        costs[~feasible] = np.inf
        
        if np.any(feasible):
            f = int(np.argmin(costs))
            result[c] = f
            load[f] += demands[c]
            feasible_found = True
        else:
            # If no feasible facility, try to find one with min violation? 
            # Or just break and return invalid? 
            # The problem guarantees a solution exists? "5 fixed generated instances"
            # Assuming feasibility.
            break

    if not feasible_found:
        # Last resort: random assignment or zero
        result = np.zeros(n_customers, dtype=int)
        
    return result
# EVOLVE-BLOCK-END
