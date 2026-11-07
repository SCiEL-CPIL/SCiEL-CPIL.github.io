# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def _compute_cost(result, demands, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open


def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    
    for i in np.argsort(-demands):
        best_f = -1
        best_cost = np.inf
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                cost = assign_cost[i, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += demands[i]
    
    # Handle any unassigned customers
    for i in range(n):
        if result[i] == -1:
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            result[i] = best_f
            load[best_f] += demands[i]
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=2000, deadline=None):
    """Local search: try moving each customer to a different facility if it improves cost."""
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = np.random.permutation(n)
        
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:  # current_f will be empty
                    delta -= open_cost[current_f]
                if load[f] == 0:  # f was empty
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if not improved:
            break
    
    return result, current_cost


def _solve_lp(demands, capacities, open_cost, assign_cost, deadline):
    """Solve as LP relaxation using min-cost flow formulation with HiGHS."""
    n = len(demands)
    m = len(capacities)
    
    # Variables: x[i,f] for each customer-facility pair (continuous [0,1])
    # y[f] for each facility (continuous [0,1])
    # Total variables: n*m + m
    total_vars = n * m + m
    
    # Objective: minimize sum(assign_cost[i,f] * x[i,f]) + sum(open_cost[f] * y[f])
    c = np.zeros(total_vars)
    for i in range(n):
        for f in range(m):
            c[i * m + f] = assign_cost[i, f]
    for f in range(m):
        c[n * m + f] = open_cost[f]
    
    # Build sparse constraint matrix
    rows = []
    cols = []
    vals = []
    
    # Constraint 1: Each customer assigned to exactly one facility: sum_f x[i,f] = 1
    for i in range(n):
        for f in range(m):
            rows.append(len(rows))
            cols.append(i * m + f)
            vals.append(1.0)
    
    # Constraint 2: Capacity: sum_i demands[i] * x[i,f] <= capacities[f]
    for f in range(m):
        for i in range(n):
            rows.append(len(rows))
            cols.append(i * m + f)
            vals.append(demands[i])
    
    # Constraint 3: Linking: x[i,f] <= y[f] for all i,f
    for i in range(n):
        for f in range(m):
            rows.append(len(rows))
            cols.append(i * m + f)
            vals.append(1.0)
            rows.append(len(rows))
            cols.append(n * m + f)
            vals.append(-1.0)
    
    A = csr_matrix((vals, (rows, cols)), shape=(len(rows), total_vars))
    
    # Bounds for each constraint
    lb = np.zeros(len(rows))
    ub = np.zeros(len(rows))
    
    # Assignment constraints: = 1
    for i in range(n):
        lb[i] = 1.0
        ub[i] = 1.0
    
    # Capacity constraints: <= capacities[f]
    for f in range(m):
        idx = n + f
        lb[idx] = 0.0
        ub[idx] = capacities[f]
    
    # Linking constraints: x[i,f] - y[f] <= 0
    for i in range(n):
        for f in range(m):
            idx = n + m + i * m + f
            lb[idx] = -np.inf
            ub[idx] = 0.0
    
    # Variable bounds: all in [0, 1]
    var_lb = np.zeros(total_vars)
    var_ub = np.ones(total_vars)
    
    remaining_time = deadline - time.time()
    if remaining_time <= 0.1:
        return None
    
    try:
        res = linprog(
            c=c,
            A_ub=A,
            b_ub=ub,
            A_eq=None,
            b_eq=None,
            bounds=list(zip(var_lb, var_ub)),
            method='highs',
            options={"time_limit": max(0.1, remaining_time), "presolve": True}
        )
        
        if res.status == 0:  # Optimal
            x_flat = res.x[:n * m]
            x = x_flat.reshape(n, m)
            
            # Round to get integer assignment
            # For each customer, assign to facility with highest x value
            result = np.argmax(x, axis=1)
            
            # Verify feasibility
            load = np.zeros(m)
            for i in range(n):
                load[result[i]] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return result
    except Exception:
        pass
    
    return None


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    # Set deadline with safety margin
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Try LP solver first
    result = _solve_lp(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is None:
        # Fallback to greedy + local search
        result = _greedy_init(demands, capacities, open_cost, assign_cost)
        result, _ = _local_search(result, demands, capacities, open_cost, assign_cost,
                                  max_iters=1000, deadline=deadline)
    
    # Final validity check
    load = np.zeros(m)
    for i in range(n):
        f = result[i]
        if f < 0 or f >= m:
            result = _greedy_init(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    if np.any(load > capacities):
        result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
