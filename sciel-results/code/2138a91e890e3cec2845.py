"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Try LP-based approach with big-M for fixed charges
    # Formulate as min-cost flow with big-M linearization
    
    # Variables:
    # x[i, j] = 1 if customer i is assigned to facility j (binary, relaxed to [0,1])
    # y[j] = 1 if facility j is open (binary, relaxed to [0,1])
    #
    # Objective: sum_{i,j} assign_cost[i,j] * x[i,j] + sum_j open_cost[j] * y[j]
    # Constraints:
    #   sum_j x[i,j] = 1 for all i (each customer assigned to exactly one facility)
    #   sum_i demands[i] * x[i,j] <= capacities[j] * y[j] for all j (capacity)
    #   x[i,j] <= y[j] for all i,j (if customer assigned, facility must be open)
    #   0 <= x[i,j] <= 1, 0 <= y[j] <= 1
    
    # Number of variables: n_customers * n_facilities (x) + n_facilities (y)
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c[n_x + j] = open_cost[j]
    
    # Equality constraints: sum_j x[i,j] = 1 for each customer i
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
    
    # Inequality constraints:
    # 1) sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0 for each facility j
    # 2) x[i,j] - y[j] <= 0 for each i,j
    
    n_ineq1 = n_facilities
    n_ineq2 = n_customers * n_facilities
    n_ineq = n_ineq1 + n_ineq2
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # Constraint type 1: capacity
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_x + j] = -capacities[j]
        b_ub[j] = 0.0
    
    # Constraint type 2: x[i,j] <= y[j]
    for i in range(n_customers):
        for j in range(n_facilities):
            A_ub[n_ineq1 + i * n_facilities + j, i * n_facilities + j] = 1.0
            A_ub[n_ineq1 + i * n_facilities + j, n_x + j] = -1.0
            b_ub[n_ineq1 + i * n_facilities + j] = 0.0
    
    # Bounds: 0 <= x[i,j] <= 1, 0 <= y[j] <= 1
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    try:
        result_lp = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result_lp.success and time.time() < deadline - 30:
            # Extract assignment from LP solution
            x_lp = result_lp.x[:n_x].reshape(n_customers, n_facilities)
            assignment = np.argmax(x_lp, axis=1)
            
            # Check validity and compute cost
            if _is_valid(assignment, demands, capacities):
                cost = _compute_cost(assignment, demands, capacities, open_cost, assign_cost)
            else:
                # LP solution may be fractional; use rounding
                assignment = _round_assignment(x_lp, demands, capacities)
                cost = _compute_cost(assignment, demands, capacities, open_cost, assign_cost)
            
            # Try local search improvement
            if time.time() < deadline - 10:
                assignment, cost = _local_search(
                    assignment, demands, capacities, open_cost, assign_cost, deadline
                )
            
            return assignment
        
    except Exception:
        pass
    
    # Fallback: greedy largest-demand-first
    return _greedy_assign(demands, capacities, assign_cost)


def _is_valid(assignment, demands, capacities):
    """Check if assignment is valid."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(assignment[i])
        if f < 0 or f >= n_facilities:
            return False
        load[f] += demands[i]
    return np.all(load <= capacities)


def _compute_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Compute total cost of assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(assignment[i])
        load[f] += demands[i]
    
    open_facilities = load > 0
    total_cost = np.sum(open_cost[open_facilities])
    for i in range(n_customers):
        f = int(assignment[i])
        total_cost += assign_cost[i, f]
    
    return total_cost


def _round_assignment(x_lp, demands, capacities):
    """Round LP solution to integer assignment using greedy on fractional values."""
    n_customers = len(x_lp)
    n_facilities = x_lp.shape[1]
    
    # Sort customers by their LP confidence (max value in their row)
    max_vals = np.max(x_lp, axis=1)
    order = np.argsort(-max_vals)
    
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Try facilities in order of LP value
        fac_order = np.argsort(-x_lp[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                assigned = True
                break
        if not assigned:
            # Find any feasible facility
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                    assigned = True
                    break
            if not assigned:
                # Should not happen if problem is feasible
                assignment[i] = 0
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """Improve assignment via single-swap local search."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    assignment = assignment.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[int(assignment[i])] += demands[i]
    
    current_cost = _compute_cost(assignment, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_assignment = assignment.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            current_f = int(assignment[i])
            # Try moving customer i to another facility
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if load[new_f] + demands[i] <= capacities[new_f]:
                    # Compute delta cost
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                    delta_open = 0.0
                    if load[current_f] == demands[i]:
                        # Current facility becomes empty
                        delta_open -= open_cost[current_f]
                    if load[new_f] == 0:
                        # New facility becomes non-empty
                        delta_open += open_cost[new_f]
                    
                    delta_total = delta_assign + delta_open
                    if delta_total < -1e-9:
                        # Accept the move
                        load[current_f] -= demands[i]
                        load[new_f] += demands[i]
                        assignment[i] = new_f
                        current_cost += delta_total
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                        break
    
    return best_assignment, best_cost


def _greedy_assign(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    order = np.argsort(-demands)
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                break
    
    return assignment
# EVOLVE-BLOCK-END
