# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _solve_milp(demands, capacities, open_cost, assign_cost, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    n_var = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c = np.zeros(n_var)
    # x_{i,f} coefficients
    c[:n_cust * n_fac] = assign_cost.flatten()
    # y_f coefficients
    c[n_cust * n_fac:] = open_cost
    
    # Constraints
    # 1. Assignment constraints: sum_f x_{i,f} = 1 for each i
    A_eq = np.zeros((n_cust, n_var))
    for i in range(n_cust):
        A_eq[i, i*n_fac:(i+1)*n_fac] = 1
    b_eq = np.ones(n_cust)
    
    # 2. Capacity constraints: sum_i demands_i * x_{i,f} - capacities_f * y_f <= 0
    A_ub = np.zeros((n_fac, n_var))
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i*n_fac + f] = demands[i]
        A_ub[f, n_cust * n_fac + f] = -capacities[f]
    b_ub = np.zeros(n_fac)
    
    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    ]
    
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary
    integrality = np.ones(n_var)
    
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": max(1.0, deadline - time.time())}
        )
        if res.status == 0: # Optimal
            x_vals = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            # Find best facility for each customer
            result = np.argmax(x_vals, axis=1)
            # Verify solution validity just in case
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[result[i]] += demands[i]
            if np.all(load <= capacities):
                return result.astype(int)
    except Exception as e:
        pass
    return None

def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1, dtype=int)
    order = np.argsort(-demands) # Largest first
    for i in order:
        # Sort facilities by assign cost
        cand = np.argsort(assign_cost[i])
        for f in cand:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    # Fallback if any unassigned
    for i in range(n_cust):
        if result[i] < 0:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    return result.astype(int)

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    start = time.time()
    deadline = start + 140.0 # Safety margin
    
    result = _solve_milp(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is None:
        result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
        
    return result
# EVOLVE-BLOCK-END
