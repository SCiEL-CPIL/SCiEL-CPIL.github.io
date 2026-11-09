"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MILP (HiGHS via scipy),
    followed by a local search polish pass.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard
    start_time = time.time()
    time_limit = 120  # total wall-clock budget
    
    # Flatten assignment variables: x[i*n_facilities + f]
    # Open variables: y[f] at offset n_customers * n_facilities
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.reshape(-1)
    c[n_x:] = open_cost
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Build constraints
    A_rows = []
    A_cols = []
    A_data = []
    
    # Assignment constraints (equality): for each customer i, sum_f x[i,f] = 1
    for i in range(n_customers):
        for f in range(n_facilities):
            A_rows.append(i)
            A_cols.append(i * n_facilities + f)
            A_data.append(1.0)
    
    # Capacity constraints (inequality: <= 0)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_rows.append(n_customers + f)
            A_cols.append(i * n_facilities + f)
            A_data.append(float(demands[i]))
        A_rows.append(n_customers + f)
        A_cols.append(n_x + f)
        A_data.append(-float(capacities[f]))
    
    A = csc_matrix((A_data, (A_rows, A_cols)), shape=(n_customers + n_facilities, n_vars))
    
    lb_con = np.zeros(n_customers + n_facilities)
    ub_con = np.zeros(n_customers + n_facilities)
    lb_con[:n_customers] = 1.0
    ub_con[:n_customers] = 1.0
    lb_con[n_customers:] = -np.inf
    ub_con[n_customers:] = 0.0
    
    constraint = LinearConstraint(A, lb_con, ub_con)
    
    # Determine time remaining for MILP solver
    elapsed = time.time() - start_time
    milp_time_limit = max(1.0, time_limit - elapsed - 15.0)  # leave 15s for local search
    
    assignments = None
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds,
                      options={"time_limit": milp_time_limit, "mip_rel_gap": 0.0})
        
        if result.status == 0 and result.x is not None:
            x_flat = result.x[:n_x]
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            assignments = np.argmax(x_matrix, axis=1)
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignments[i]] += demands[i]
            
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-6:
                    assignments = None
                    break
        else:
            assignments = None
            
    except Exception:
        assignments = None
    
    if assignments is None:
        assignments = _greedy_fallback(demands, capacities, assign_cost)
    
    # Local search polish pass
    elapsed = time.time() - start_time
    local_search_time = max(0.5, time_limit - elapsed - 5.0)
    if local_search_time > 0.5:
        assignments = _local_search(demands, capacities, open_cost, assign_cost, assignments,
                                     time_budget=local_search_time)
    
    return assignments.astype(int)


def _local_search(demands, capacities, open_cost, assign_cost, assignments, time_budget=10.0):
    """
    Local search: try reassigning each customer to a different facility.
    Accept if total cost decreases and capacity constraints are maintained.
    Uses first-improvement with multiple passes until no improvement or time runs out.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    
    # Compute current total cost
    def total_cost(assign):
        cost = np.sum(assign_cost[np.arange(n_customers), assign])
        used = set(assign.tolist())
        cost += np.sum(open_cost[list(used)])
        return cost
    
    current_cost = total_cost(assignments)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[assignments[i]] += demands[i]
    
    improved = True
    while improved:
        improved = False
        for i in range(n_customers):
            # Check time
            if time.time() - start_time > time_budget:
                return assignments
            
            current_f = assignments[i]
            # Try all other facilities
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                
                # Check capacity: removing i from current_f, adding to new_f
                new_load_new = load[new_f] + demands[i]
                if new_load_new > capacities[new_f]:
                    continue
                
                # Compute cost delta
                # Change in assignment cost
                delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                # Change in open cost:
                # If current_f becomes empty, we save open_cost[current_f]
                # If new_f was empty, we add open_cost[new_f]
                old_load_current = load[current_f] - demands[i]
                old_load_new = load[new_f]
                
                delta_open = 0.0
                if old_load_current == 0:
                    delta_open -= open_cost[current_f]
                if old_load_new == 0:
                    delta_open += open_cost[new_f]
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Accept move
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    assignments[i] = new_f
                    current_cost += delta
                    improved = True
                    break  # move to next customer
    
    return assignments


def _greedy_fallback(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        sorted_f = np.argsort(assign_cost[i])
        for f in sorted_f:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    
    return result
# EVOLVE-BLOCK-END
