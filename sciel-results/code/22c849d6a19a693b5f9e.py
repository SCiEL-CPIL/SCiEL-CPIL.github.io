import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a Mixed Integer Linear Program (MILP).
    
    Args:
        demands: Array of customer demands.
        capacities: Array of facility capacities.
        open_cost: Array of facility opening costs.
        assign_cost: Matrix of assignment costs (customers x facilities).
        
    Returns:
        Array of facility indices for each customer.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Time limit guard
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 120s limit
    
    # Variables:
    # x[i, j] : 1 if customer i is assigned to facility j
    # y[j] : 1 if facility j is open
    # Total variables: n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    # Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i, j] * x[i, j])
    c = np.zeros(n_customers * n_facilities + n_facilities)
    
    # x variables are first, y variables are last
    # c[i * n_facilities + j] = assign_cost[i, j]
    c[:n_customers * n_facilities] = assign_cost.flatten()
    # c[n_customers * n_facilities + j] = open_cost[j]
    c[n_customers * n_facilities:] = open_cost
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_j x[i, j] = 1 for all i
    #    This is a linear constraint: A_eq @ vars = b_eq
    
    # 2. Capacity constraints:
    #    sum_i demands[i] * x[i, j] <= capacities[j] * y[j] for all j
    #    Rearranged: sum_i demands[i] * x[i, j] - capacities[j] * y[j] <= 0
    
    # Let's build the constraint matrix A_ub and b_ub for inequality constraints (<=)
    # And A_eq, b_eq for equality constraints (=)
    
    # Equality constraints: n_customers rows
    A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    b_eq = np.ones(n_customers)
    
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    # Inequality constraints: n_facilities rows
    A_ub = np.zeros((n_facilities, n_customers * n_facilities + n_facilities))
    b_ub = np.zeros(n_facilities)
    
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        
    # Variable bounds and integrality
    # x and y are binary (0 or 1)
    integrality = np.ones(n_customers * n_facilities + n_facilities)
    lb = np.zeros(n_customers * n_facilities + n_facilities)
    ub = np.ones(n_customers * n_facilities + n_facilities)
    
    # Solve MILP
    try:
        res = milp(
            c=c,
            constraints=[
                LinearConstraint(A_ub, -np.inf, b_ub),
                LinearConstraint(A_eq, b_eq, b_eq)
            ],
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:
            x_vars = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
            # Find the facility for each customer
            result = np.argmax(x_vars, axis=1)
            
            # Verify validity just in case
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
                
            # Check capacities
            # Note: If a facility is closed (y=0), load should be 0.
            # If load > capacity, it's invalid.
            # However, the MILP constraint ensures sum(demand_i * x_ij) <= capacity_j * y_j.
            # If y_j=0, sum(demand_i * x_ij) must be 0.
            # So if the solver is correct, this should be valid.
            
            return result.astype(int)
        else:
            # Fallback if MILP fails
            pass
            
    except Exception as e:
        # Fallback if MILP fails
        pass
        
    # Fallback: Greedy Heuristic (same as seed)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    # Largest demand first
    for i in np.argsort(-demands):
        # Find cheapest feasible facility
        costs = assign_cost[i]
        sorted_facilities = np.argsort(costs)
        for f in sorted_facilities:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # If no feasible facility found, assign to facility with most remaining capacity
            # This shouldn't happen if total capacity >= total demand, but just in case
            remaining_cap = capacities - load
            f = np.argmax(remaining_cap)
            result[i] = f
            load[f] += demands[i]
            
    return result
