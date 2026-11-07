"""Improved program using Mixed Integer Linear Programming (MIP) for exact Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed Integer Linear Programming (MIP).
    This finds the exact optimal solution for the given instance size.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if n_customers == 0:
        return np.array([], dtype=int)

    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)

    # Variables:
    # x[i, j] for i in 0..n_customers-1, j in 0..n_facilities-1
    # y[j] for j in 0..n_facilities-1
    # Total variables: n_customers * n_facilities + n_facilities
    
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c[n_customers * n_facilities + j] = open_cost[j]

    # Bounds: All variables are binary [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: All variables are integer (binary)
    integrality = np.ones(n_vars)
    
    # Constraints:
    # 1. Each customer is assigned to exactly one facility:
    #    sum_j x[i, j] = 1 for each i
    #    This gives n_customers equality constraints.
    
    # 2. Capacity constraints:
    #    sum_i demands[i] * x[i, j] <= capacities[j] * y[j] for each j
    #    sum_i demands[i] * x[i, j] - capacities[j] * y[j] <= 0
    #    This gives n_facilities inequality constraints.
    
    # 3. Linking constraints:
    #    x[i, j] <= y[j] for each i, j
    #    x[i, j] - y[j] <= 0
    #    This gives n_customers * n_facilities inequality constraints.
    
    n_cons_eq = n_customers
    n_cons_cap = n_facilities
    n_cons_link = n_customers * n_facilities
    total_cons = n_cons_eq + n_cons_cap + n_cons_link
    
    A_eq = np.zeros((n_cons_eq, n_vars))
    b_eq = np.ones(n_cons_eq)
    
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    A_ub = np.zeros((n_cons_cap + n_cons_link, n_vars))
    b_ub = np.zeros(n_cons_cap + n_cons_link)
    
    # Fill capacity constraints (rows 0 to n_facilities-1)
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        b_ub[j] = 0.0
        
    # Fill linking constraints (rows n_facilities to end)
    row_offset = n_facilities
    for i in range(n_customers):
        for j in range(n_facilities):
            idx = row_offset + i * n_facilities + j
            A_ub[idx, i * n_facilities + j] = 1.0
            A_ub[idx, n_customers * n_facilities + j] = -1.0
            b_ub[idx] = 0.0

    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    ]
    
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_customers * n_facilities]
            # Reshape to (n_customers, n_facilities)
            x = x_flat.reshape((n_customers, n_facilities))
            
            # Find the facility for each customer
            # Due to floating point, argmax might be ambiguous if multiple 1s, but MIP should give 1.0
            # We take the index where value is closest to 1
            assignment = np.argmax(x, axis=1)
            
            return assignment.astype(int)
        else:
            # If MIP fails, fallback to greedy
            return _fallback_greedy(demands, capacities, open_cost, assign_cost)
            
    except Exception:
        # In case of any error, use fallback
        return _fallback_greedy(demands, capacities, open_cost, assign_cost)

def _fallback_greedy(demands, capacities, open_cost, assign_cost):
    """Simple greedy fallback to ensure validity."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    order = np.argsort(-demands)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                if assign_cost[i, f] < best_cost:
                    best_cost = assign_cost[i, f]
                    best_f = f
        if best_f == -1:
            # Should not happen if feasible
            best_f = np.argmax(capacities - load)
        assignment[i] = best_f
        load[best_f] += demands[i]
        
    return assignment
# EVOLVE-BLOCK-END
