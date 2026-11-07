import numpy as np
import time
from scipy.optimize import linprog

def _solve_assignment(demands, capacities, assign_cost, open_facilities):
    """
    Solve the assignment problem for a fixed set of open facilities.
    Returns (assignment_array, assignment_cost) or (None, inf) if infeasible.
    """
    n_customers = len(demands)
    n_open = len(open_facilities)
    
    if n_open == 0:
        return None, np.inf

    # Map open facility indices to 0..n_open-1 for the LP
    # x[i, j] where i is customer, j is index in open_facilities
    n_vars = n_customers * n_open
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for k, j in enumerate(open_facilities):
            c[i * n_open + k] = assign_cost[i, j]
            
    # Constraints
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for k in range(n_open):
            A_eq[i, i * n_open + k] = 1.0
            
    A_ub = np.zeros((n_open, n_vars))
    b_ub = np.zeros(n_open)
    for k, j in enumerate(open_facilities):
        for i in range(n_customers):
            A_ub[k, i * n_open + k] = demands[i]
        b_ub[k] = capacities[j]
        
    bounds = [(0, 1) for _ in range(n_vars)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        return None, np.inf
        
    if res.status != 0:
        return None, np.inf
        
    x = res.x.reshape(n_customers, n_open)
    
    # Extract assignment
    assignment = np.zeros(n_customers, dtype=int)
    total_cost = 0.0
    for i in range(n_customers):
        # Find the max x (should be 1.0)
        j_idx = np.argmax(x[i])
        # Check if it's close to 1
        if x[i, j_idx] < 0.99:
             # Fallback: pick min cost among those with x > 0.5
             candidates = np.where(x[i] > 0.5)[0]
             if len(candidates) == 0:
                 return None, np.inf
             # Get actual facility indices
             cand_facs = [open_facilities[k] for k in candidates]
             # Pick min cost
             costs = [assign_cost[i, f] for f in cand_facs]
             j_idx = candidates[np.argmin(costs)]
        
        assignment[i] = open_facilities[j_idx]
        total_cost += assign_cost[i, assignment[i]]
        
    return assignment, total_cost

def _greedy_init(demands, capacities, assign_cost):
    """
    Greedy initialization to find a feasible set of open facilities.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    load = np.zeros(n_facilities)
    assignment = np.full(n_customers, -1, dtype=int)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities
        feasible = [j for j in range(n_facilities) if load[j] + d <= capacities[j]]
        if not feasible:
            # No feasible facility, pick one with max remaining capacity
            j = np.argmax(capacities - load)
            feasible = [j]
            
        # Among feasible, pick min cost
        costs = [assign_cost[i, j] for j in feasible]
        best_j = feasible[np.argmin(costs)]
        assignment[i] = best_j
        load[best_j] += d
        
    # Determine open facilities
    open_set = set(assignment)
    return sorted(list(open_set)), assignment

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Initialization
    current_open_list, current_assignment = _greedy_init(demands, capacities, assign_cost)
    
    # 2. Polish Assignment for the Initial Set
    current_assignment, current_assign_cost = _solve_assignment(demands, capacities, assign_cost, current_open_list)
    
    if current_assignment is None:
        # Fallback to greedy assignment directly if LP fails
        current_assignment = _greedy_init(demands, capacities, assign_cost)[1]
        current_open_list = sorted(set(current_assignment))
        current_assign_cost = np.sum([assign_cost[i, current_assignment[i]] for i in range(n_customers)])
        
    def get_total_cost(open_list, assign_cost_val):
        open_cost_val = np.sum([open_cost[j] for j in open_list])
        return open_cost_val + assign_cost_val
        
    current_total_cost = get_total_cost(current_open_list, current_assign_cost)
    best_assignment = current_assignment.copy()
    best_total_cost = current_total_cost
    best_open_list = current_open_list.copy()
    
    # 3. Local Search on Facility Set
    # We will try to improve the set of open facilities.
    # Moves: Add one, Remove one.
    # If time permits, Swap one.
    
    max_iterations = 100
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try Adding a facility
        best_move = None
        best_move_cost = current_total_cost
        best_move_type = None
        best_move_args = None
        
        # Candidate: Add j
        for j in range(n_facilities):
            if j in current_open_list:
                continue
            new_open_list = current_open_list + [j]
            new_open_list.sort()
            a, c = _solve_assignment(demands, capacities, assign_cost, new_open_list)
            if a is not None:
                tc = get_total_cost(new_open_list, c)
                if tc < best_move_cost - 1e-6:
                    best_move_cost = tc
                    best_move_type = 'add'
                    best_move_args = (new_open_list, a, c)
                    
        # Candidate: Remove j
        for j in current_open_list:
            if len(current_open_list) == 1:
                continue # Can't remove last one
            new_open_list = [x for x in current_open_list if x != j]
            a, c = _solve_assignment(demands, capacities, assign_cost, new_open_list)
            if a is not None:
                tc = get_total_cost(new_open_list, c)
                if tc < best_move_cost - 1e-6:
                    best_move_cost = tc
                    best_move_type = 'remove'
                    best_move_args = (new_open_list, a, c)
                    
        if best_move_type is not None:
            new_open_list, a, c = best_move_args
            current_open_list = new_open_list
            current_assignment = a
            current_assign_cost = c
            current_total_cost = best_move_cost
            
            # Update best
            if current_total_cost < best_total_cost:
                best_total_cost = current_total_cost
                best_assignment = current_assignment.copy()
                best_open_list = current_open_list.copy()
                
            improved = True
        else:
            # No add/remove improvement. Try swaps if we have time.
            if time.time() - start_time < timeout - 10.0:
                # Try Swaps
                for j in current_open_list:
                    for k in range(n_facilities):
                        if k in current_open_list:
                            continue
                        new_open_list = [x for x in current_open_list if x != j] + [k]
                        new_open_list.sort()
                        a, c = _solve_assignment(demands, capacities, assign_cost, new_open_list)
                        if a is not None:
                            tc = get_total_cost(new_open_list, c)
                            if tc < best_move_cost - 1e-6:
                                best_move_cost = tc
                                best_move_type = 'swap'
                                best_move_args = (new_open_list, a, c)
                                break
                    if best_move_type:
                        break
                
                if best_move_type is not None:
                    new_open_list, a, c = best_move_args
                    current_open_list = new_open_list
                    current_assignment = a
                    current_assign_cost = c
                    current_total_cost = best_move_cost
                    
                    # Update best
                    if current_total_cost < best_total_cost:
                        best_total_cost = current_total_cost
                        best_assignment = current_assignment.copy()
                        best_open_list = current_open_list.copy()
                    
                    improved = True
        
        if not improved:
            break
            
        if time.time() - start_time > timeout - 5.0:
            break

    return best_assignment
