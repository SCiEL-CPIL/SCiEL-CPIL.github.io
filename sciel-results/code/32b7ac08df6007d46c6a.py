"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MILP (HiGHS via scipy).
    Uses time budget with local search polish.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Flatten assignment variables: x[i*n_facilities + f]
    # Open variables: y[f] at offset n_customers * n_facilities
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.ravel()
    c[n_x:] = open_cost
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Build constraint matrix efficiently
    A_rows = []
    A_cols = []
    A_data = []
    
    # Assignment constraints (equality): for each customer i, sum_f x[i,f] = 1
    for i in range(n_customers):
        row = i
        cols = np.arange(i * n_facilities, (i + 1) * n_facilities)
        A_rows.extend([row] * n_facilities)
        A_cols.extend(cols.tolist())
        A_data.extend([1.0] * n_facilities)
    
    # Capacity constraints (inequality: <= 0)
    # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    for f in range(n_facilities):
        row = n_customers + f
        for i in range(n_customers):
            A_rows.append(row)
            A_cols.append(i * n_facilities + f)
            A_data.append(float(demands[i]))
        A_rows.append(row)
        A_cols.append(n_x + f)
        A_data.append(-float(capacities[f]))
    
    A = csc_matrix((A_data, (A_rows, A_cols)), shape=(n_customers + n_facilities, n_vars))
    
    # Lower and upper bounds for constraints
    lb_con = np.zeros(n_customers + n_facilities)
    ub_con = np.zeros(n_customers + n_facilities)
    
    # Assignment constraints: equality to 1
    lb_con[:n_customers] = 1.0
    ub_con[:n_customers] = 1.0
    
    # Capacity constraints: <= 0
    lb_con[n_customers:] = -np.inf
    ub_con[n_customers:] = 0.0
    
    constraint = LinearConstraint(A, lb_con, ub_con)
    
    best_assignment = None
    best_cost = np.inf
    
    # Solve MILP
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            x_flat = result.x[:n_x]
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            assignments = np.argmax(x_matrix, axis=1)
            
            # Verify validity
            if _is_valid(assignments, demands, capacities):
                best_assignment = assignments.astype(int)
                best_cost = _calc_cost(best_assignment, demands, capacities, open_cost, assign_cost)
        else:
            # Fallback to greedy
            best_assignment = _greedy_fallback(demands, capacities, assign_cost)
            best_cost = _calc_cost(best_assignment, demands, capacities, open_cost, assign_cost)
            
    except Exception:
        # Fallback to greedy
        best_assignment = _greedy_fallback(demands, capacities, assign_cost)
        best_cost = _calc_cost(best_assignment, demands, capacities, open_cost, assign_cost)
    
    # Local search polish with remaining time
    elapsed = time.time() - start_time
    if best_assignment is not None and elapsed < time_limit - 10:
        polished, polished_cost = _local_search(
            best_assignment, demands, capacities, open_cost, assign_cost,
            time_limit - elapsed - 5, start_time
        )
        if polished_cost < best_cost:
            best_assignment = polished
            best_cost = polished_cost
    
    return best_assignment


def _is_valid(assignments, demands, capacities):
    """Check if assignment is valid (all customers assigned, no capacity violations)."""
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i, f in enumerate(assignments):
        if f < 0 or f >= n_facilities:
            return False
        load[f] += demands[i]
    return np.all(load <= capacities + 1e-6)


def _calc_cost(assignments, demands, capacities, open_cost, assign_cost):
    """Calculate total cost for an assignment."""
    n_facilities = len(capacities)
    open_facs = np.zeros(n_facilities, dtype=bool)
    for f in assignments:
        open_facs[f] = True
    open_cost_total = np.sum(open_cost[open_facs])
    assign_cost_total = np.sum(assign_cost[np.arange(len(assignments)), assignments])
    return open_cost_total + assign_cost_total


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
            # Should not happen if capacities are sufficient, but assign to first available
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    
    return result


def _local_search(assignments, demands, capacities, open_cost, assign_cost, time_budget, start_time):
    """
    Local search: try moving each customer to a different facility.
    Also try swapping pairs of customers.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    current = assignments.copy().astype(int)
    current_cost = _calc_cost(current, demands, capacities, open_cost, assign_cost)
    
    load = np.zeros(n_facilities)
    for i, f in enumerate(current):
        load[f] += demands[i]
    
    open_facs = np.zeros(n_facilities, dtype=bool)
    for f in current:
        open_facs[f] = True
    
    improved = True
    while improved and (time.time() - start_time) < start_time + time_budget:
        improved = False
        
        # Try single moves
        for i in range(n_customers):
            if (time.time() - start_time) >= start_time + time_budget:
                break
            old_f = current[i]
            for new_f in range(n_facilities):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Calculate delta cost
                delta = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                # If old facility becomes empty, we save its open cost
                if load[old_f] == demands[i] and open_facs[old_f]:
                    delta -= open_cost[old_f]
                
                # If new facility was closed, we pay its open cost
                if not open_facs[new_f]:
                    delta += open_cost[new_f]
                
                if delta < -1e-9:
                    # Accept the move
                    load[old_f] -= demands[i]
                    if load[old_f] == 0:
                        open_facs[old_f] = False
                    load[new_f] += demands[i]
                    open_facs[new_f] = True
                    current[i] = new_f
                    current_cost += delta
                    improved = True
                    break
        
        # Try pair swaps
        if not improved:
            for i in range(n_customers):
                if (time.time() - start_time) >= start_time + time_budget:
                    break
                for j in range(i + 1, n_customers):
                    if (time.time() - start_time) >= start_time + time_budget:
                        break
                    fi = current[i]
                    fj = current[j]
                    if fi == fj:
                        continue
                    
                    # Check if swap is feasible
                    # After swap: i goes to fj, j goes to fi
                    # Load at fi: -demands[i] + demands[j]
                    # Load at fj: -demands[j] + demands[i]
                    if load[fi] - demands[i] + demands[j] > capacities[fi]:
                        continue
                    if load[fj] - demands[j] + demands[i] > capacities[fj]:
                        continue
                    
                    # Calculate delta cost
                    delta = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                            (assign_cost[i, fi] + assign_cost[j, fj])
                    
                    # Open cost changes:
                    # If fi becomes empty after removing i, save open_cost[fi]
                    if load[fi] == demands[i] and open_facs[fi]:
                        delta -= open_cost[fi]
                    # If fj becomes empty after removing j, save open_cost[fj]
                    if load[fj] == demands[j] and open_facs[fj]:
                        delta -= open_cost[fj]
                    # If fi was closed but now has j, pay open_cost[fi]
                    if not open_facs[fi]:
                        delta += open_cost[fi]
                    # If fj was closed but now has i, pay open_cost[fj]
                    if not open_facs[fj]:
                        delta += open_cost[fj]
                    
                    if delta < -1e-9:
                        # Accept the swap
                        load[fi] -= demands[i]
                        load[fj] -= demands[j]
                        if load[fi] == 0:
                            open_facs[fi] = False
                        if load[fj] == 0:
                            open_facs[fj] = False
                        load[fj] += demands[i]
                        load[fi] += demands[j]
                        open_facs[fj] = True
                        open_facs[fi] = True
                        current[i] = fj
                        current[j] = fi
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
    
    return current, current_cost
# EVOLVE-BLOCK-END
