"""Capacitated facility location solver using MILP via HiGHS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: leave ~2s margin before the 150s wall-clock limit
    start_time = time.time()
    time_limit = 145.0  # seconds
    
    # Decision variables:
    # x[i, f] in {0,1}: customer i assigned to facility f
    # y[f] in {0,1}: facility f is open
    #
    # Total variables: n_cust * n_fac + n_fac
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective: minimize sum(open_cost[f] * y[f]) + sum(assign_cost[i,f] * x[i,f])
    c = np.zeros(n_vars)
    for f in range(n_fac):
        c[n_x + f] = open_cost[f]
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    
    # Constraints:
    # 1) Each customer assigned to exactly one facility:
    #    sum_f x[i,f] = 1 for each i
    # 2) Capacity: sum_i demands[i] * x[i,f] <= capacities[f] for each f
    # 3) Linking: x[i,f] <= y[f] for all i,f (if customer i uses facility f, f must be open)
    
    n_constraints = n_cust + n_fac + n_cust * n_fac
    
    A = np.zeros((n_constraints, n_vars))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    # Constraint 1: assignment (equality)
    for i in range(n_cust):
        row = i
        for f in range(n_fac):
            A[row, i * n_fac + f] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
    
    # Constraint 2: capacity
    for f in range(n_fac):
        row = n_cust + f
        for i in range(n_cust):
            A[row, i * n_fac + f] = demands[i]
        lb[row] = 0.0
        ub[row] = capacities[f]
    
    # Constraint 3: linking x[i,f] <= y[f]  =>  x[i,f] - y[f] <= 0
    for i in range(n_cust):
        for f in range(n_fac):
            row = n_cust + n_fac + i * n_fac + f
            A[row, i * n_fac + f] = 1.0
            A[row, n_x + f] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
    
    # Variable bounds: all in [0,1]
    var_lb = np.zeros(n_vars)
    var_ub = np.ones(n_vars)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(n_vars)
    
    # Check time before solving
    if time.time() - start_time > time_limit:
        # Fallback: greedy
        return _greedy_fallback(demands, capacities, assign_cost)
    
    # Solve MILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub),
            options={"time_limit": time_limit - (time.time() - start_time) - 1.0}
        )
    except Exception:
        return _greedy_fallback(demands, capacities, assign_cost)
    
    if time.time() - start_time > time_limit:
        return _greedy_fallback(demands, capacities, assign_cost)
    
    if not res.success:
        # Try to use best feasible solution if available
        if res.x is not None and res.status == 2:  # 2 = best solution found within time limit
            pass  # use res.x
        else:
            return _greedy_fallback(demands, capacities, assign_cost)
    
    # Extract assignment from x variables
    x_vals = res.x[:n_x].reshape(n_cust, n_fac)
    result = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        # Find facility with highest x value (should be 1 for the assigned one)
        f = int(np.argmax(x_vals[i]))
        result[i] = f
    
    # Validate and fix if needed
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Check capacity violations
    if np.any(load > capacities + 1e-6):
        return _greedy_fallback(demands, capacities, assign_cost)
    
    return result


def _greedy_fallback(demands, capacities, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                best_f = f
                break
        if best_f == -1:
            # Should not happen, but assign to least-loaded feasible
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                best_f = int(np.argmin(load))
        load[best_f] += demands[i]
        result[i] = best_f
    return result
# EVOLVE-BLOCK-END
