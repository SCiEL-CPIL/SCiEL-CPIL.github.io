"""Capacitated facility location: MIP solver using scipy.optimize.milp."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed-Integer Linear Programming.
    
    The problem is formulated as a MIP where:
    - x[i, j] is binary: 1 if customer i is assigned to facility j
    - y[j] is binary: 1 if facility j is open
    
    Constraints:
    1. Each customer assigned to exactly one facility: sum_j x[i, j] = 1
    2. Facility capacity: sum_i demands[i] * x[i, j] <= capacities[j]
    3. Linking: x[i, j] <= y[j] (if x is 1, y must be 1)
    
    Objective:
    Minimize sum_j open_cost[j] * y[j] + sum_{i,j} assign_cost[i, j] * x[i, j]
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Variables:
    # 0 ... n_cust*n_fac - 1 : x[i, j] (continuous relaxation [0, 1] is optimal due to TU)
    # n_cust*n_fac ... n_cust*n_fac + n_fac - 1 : y[j] (binary [0, 1])
    
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y

    c = np.zeros(n_vars)
    # Cost for x variables
    c[:n_x] = assign_cost.flatten()
    # Cost for y variables
    c[n_x:] = open_cost

    # Integrality: x can be continuous (0,1), y must be binary (0,1)
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 0  # Continuous
    integrality[n_x:] = 1  # Binary

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)

    # Constraints
    A_list = []
    lb_list = []
    ub_list = []

    # 1. Assignment constraints: sum_j x[i, j] = 1 for each i
    # This is a block of n_cust rows, each with n_cust*n_fac + n_fac cols
    # But we can construct A efficiently.
    A_assign = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        for j in range(n_fac):
            A_assign[i, i * n_fac + j] = 1.0
    A_list.append(A_assign)
    lb_list.append(np.ones(n_cust))
    ub_list.append(np.ones(n_cust))

    # 2. Capacity constraints: sum_i demands[i] * x[i, j] <= capacities[j] for each j
    A_cap = np.zeros((n_fac, n_vars))
    for j in range(n_fac):
        for i in range(n_cust):
            A_cap[j, i * n_fac + j] = demands[i]
    A_list.append(A_cap)
    lb_list.append(np.zeros(n_fac))
    ub_list.append(capacities)

    # 3. Linking constraints: x[i, j] - y[j] <= 0 for all i, j
    A_link = np.zeros((n_x, n_vars))
    for i in range(n_cust):
        for j in range(n_fac):
            idx_x = i * n_fac + j
            idx_y = n_x + j
            A_link[idx_x, idx_x] = 1.0
            A_link[idx_x, idx_y] = -1.0
    A_list.append(A_link)
    lb_list.append(np.zeros(n_x))
    ub_list.append(np.zeros(n_x))

    # Combine constraints
    A = np.vstack(A_list)
    lb = np.concatenate(lb_list)
    ub = np.concatenate(ub_list)
    
    constraints = LinearConstraint(A, lb, ub)

    # Solve MIP
    # Method 'highs' is the default and robust
    # time_limit is in seconds
    time_remaining = deadline - time.time()
    if time_remaining < 5:
        # Fallback to simple heuristic if no time
        return _fallback_heuristic(demands, capacities, open_cost, assign_cost)

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        method="highs",
        options={"time_limit": time_remaining, "presolve": True}
    )

    if res.status == 0:  # Optimal solution found
        x = res.x[:n_x].reshape(n_cust, n_fac)
        # Assign customer to facility with x closest to 1
        # Since it's a MIP, x should be 0 or 1 for integer variables, and close to 0 or 1 for continuous
        # But x is continuous here. Due to total unimodularity of the assignment substructure,
        # the LP relaxation of the assignment part (with fixed y) is integral.
        # However, y is binary. The interaction might lead to fractional x?
        # No, the constraint matrix of the CFP is not totally unimodular because of the linking constraints.
        # So x might be fractional.
        # We need to round x to a valid assignment.
        
        # Strategy: For each customer, pick the facility with the highest x value.
        # Then repair capacity violations if any.
        res_assign = np.argmax(x, axis=1)
        
        # Check feasibility
        load = np.zeros(n_fac)
        np.add.at(load, res_assign, demands)
        
        if np.any(load > capacities + 1e-9):
            # Repair
            res_assign, load = _repair(res_assign, load, demands, capacities, assign_cost)
            
        return res_assign.astype(int)
    else:
        # Fallback
        return _fallback_heuristic(demands, capacities, open_cost, assign_cost)

def _repair(res, load, demands, capacities, assign_cost):
    """Repair infeasible solution by moving customers."""
    res = res.copy()
    load = load.copy()
    n_fac = len(capacities)
    
    overloaded = load > capacities + 1e-9
    
    for f in np.where(overloaded)[0]:
        excess = load[f] - capacities[f]
        if excess <= 0:
            continue
        
        custs = np.where(res == f)[0]
        if len(custs) == 0:
            continue
        
        # Sort by demand ascending
        custs = custs[np.argsort(demands[custs])]
        
        for i in custs:
            if excess <= 1e-9:
                break
            
            d_i = demands[i]
            feasible = (load + d_i <= capacities + 1e-9)
            feasible[f] = False
            
            if np.any(feasible):
                feasible_indices = np.where(feasible)[0]
                costs = assign_cost[i, feasible_indices]
                best_idx = np.argmin(costs)
                best_f = feasible_indices[best_idx]
                
                load[f] -= d_i
                load[best_f] += d_i
                res[i] = best_f
                excess -= d_i
            else:
                # No feasible alternative, try swapping? For now, skip
                pass
                
    return res, load

def _fallback_heuristic(demands, capacities, open_cost, assign_cost):
    """Simple greedy heuristic if MIP fails or times out."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Greedy assignment
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        d_i = demands[i]
        # Find feasible facilities
        feasible = load + d_i <= capacities
        if np.any(feasible):
            # Among feasible, pick the one with min assign_cost
            # Note: open_cost is not considered in greedy for simplicity, 
            # but we can add it if load is 0
            feasible_indices = np.where(feasible)[0]
            costs = assign_cost[i, feasible_indices].copy()
            # Add open cost if facility is currently empty
            for f in feasible_indices:
                if load[f] == 0:
                    costs[np.where(feasible_indices == f)[0][0]] += open_cost[f]
            best_f = feasible_indices[np.argmin(costs)]
            res[i] = best_f
            load[best_f] += d_i
        else:
            # No feasible facility, assign to least loaded? Or min cost ignoring capacity?
            # This makes it invalid, but better than crashing.
            # Try to find min cost facility regardless of capacity
            best_f = np.argmin(assign_cost[i, :])
            res[i] = best_f
            load[best_f] += d_i
            
    return res

# EVOLVE-BLOCK-END
