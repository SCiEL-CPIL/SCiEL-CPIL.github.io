"""Solve capacitated facility location via MILP (HiGHS) with greedy fallback."""
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
            # Should not happen if total capacity >= total demand, but safety
            result[i] = 0
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
    return _greedy_fallback(demands, capacities, assign_cost)
# EVOLVE-BLOCK-END
