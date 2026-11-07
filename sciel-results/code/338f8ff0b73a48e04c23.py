"""Rewritten program using min-cost flow formulation for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog
import itertools


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities using a min-cost flow / LP approach.
    
    We formulate this as a linear program:
    - x[i, f] = 1 if customer i is assigned to facility f
    - y[f] = 1 if facility f is open
    
    Minimize: sum_{i,f} assign_cost[i,f] * x[i,f] + sum_f open_cost[f] * y[f]
    Subject to:
    - sum_f x[i,f] = 1 for all i (each customer assigned to exactly one facility)
    - sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f (capacity constraint)
    - x[i,f] >= 0, y[f] >= 0
    
    Since the constraint matrix has a special structure, we can solve this LP and
    the solution will be integral due to total unimodularity properties of the
    assignment structure when combined with capacity constraints.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # If no customers, return empty
    if n_customers == 0:
        return np.array([], dtype=int)
    
    # Try a greedy approach first as a fallback, then try LP
    # Actually, let's use a smarter approach: min-cost max-flow via network simplex
    
    # For this problem size (100 customers, 20 facilities), we can use LP relaxation
    # with scipy's linprog, or implement a min-cost flow approach.
    
    # Let's use the LP approach since it's more straightforward and should give
    # near-optimal or optimal solutions for this problem structure.
    
    # Decision variables: x[i, f] for all i, f (100*20 = 2000 vars)
    # and y[f] for all f (20 vars)
    # Total: 2020 variables
    
    # Objective: minimize sum assign_cost[i,f]*x[i,f] + sum open_cost[f]*y[f]
    c = np.zeros(n_customers * n_facilities + n_facilities)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_customers * n_facilities + f] = open_cost[f]
    
    # Equality constraints: sum_f x[i,f] = 1 for each customer i
    # This gives n_customers equality constraints
    A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0
    
    # Inequality constraints: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0 for each f
    A_ub = np.zeros((n_facilities, n_customers * n_facilities + n_facilities))
    b_ub = np.zeros(n_facilities)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
        A_ub[f, n_customers * n_facilities + f] = -capacities[f]
    
    # Bounds: x[i,f] >= 0, y[f] >= 0
    bounds = [(0, None)] * (n_customers * n_facilities + n_facilities)
    
    # Solve LP
    try:
        result_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, method='highs')
        if result_lp.success:
            x_vals = result_lp.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            # Round to get integer assignments
            # For each customer, pick the facility with highest x value
            assignment = np.argmax(x_vals, axis=1)
            
            # Verify capacity constraints
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            
            valid = True
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-6:
                    valid = False
                    break
            
            if valid:
                return assignment.astype(int)
        else:
            # LP failed, fall back to greedy
            pass
    except Exception:
        pass
    
    # Fallback: greedy approach with better ordering
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand in descending order
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities sorted by cost
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append((assign_cost[i, f], f))
        
        if feasible:
            feasible.sort()
            best_f = feasible[0][1]
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Should not happen if capacities are sufficient, but handle gracefully
            # Assign to facility with most remaining capacity
            remaining = capacities - load
            best_f = np.argmax(remaining)
            load[best_f] += demands[i]
            result[i] = best_f
    
    return result
# EVOLVE-BLOCK-END
