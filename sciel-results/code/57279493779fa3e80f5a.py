"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _solve_lp(demands, capacities, open_cost, assign_cost):
    """
    Solve the LP relaxation of the Capacitated Facility Location Problem.
    Variables:
    x_{i,j} in [0,1] for i in customers, j in facilities
    y_j in [0,1] for j in facilities
    
    Objective: Minimize sum_{i,j} assign_cost[i,j]*x_{i,j} + sum_j open_cost[j]*y_j
    
    Constraints:
    1. sum_j x_{i,j} = 1 for all i
    2. sum_i demands[i]*x_{i,j} <= capacities[j] for all j
    3. x_{i,j} - y_j <= 0 for all i,j
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    n_x = n_cust * n_fac
    n_vars = n_x + n_fac
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c[n_x + j] = open_cost[j]
        
    # Constraints in A_ub @ vars <= b_ub form
    # We have equality constraints for assignment, which we can handle by 
    # creating two inequalities or by using the bounds if supported, but 
    # linprog standard form is A_ub x <= b_ub and A_eq x = b_eq.
    
    # A_eq for assignment: sum_j x_{i,j} = 1
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
            
    # A_ub for capacity and linking
    n_ineq = n_fac + n_x
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    # Capacity constraints
    for j in range(n_fac):
        for i in range(n_cust):
            A_ub[row, i * n_fac + j] = demands[i]
        b_ub[row] = capacities[j]
        row += 1
        
    # Linking constraints: x_{i,j} - y_j <= 0
    for i in range(n_cust):
        for j in range(n_fac):
            A_ub[row, i * n_fac + j] = 1.0
            A_ub[row, n_x + j] = -1.0
            # b_ub[row] is already 0
            row += 1
            
    # Bounds
    bounds = [(0, 1)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            x_flat = res.x[:n_x]
            y_flat = res.x[n_x:]
            
            # Round x to get a valid assignment
            # Since it's an LP, x might be fractional.
            # We need a valid integer assignment.
            # A simple heuristic: for each customer, pick the facility with max x_{i,j}.
            # If ties or issues, fall back to greedy.
            result = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            
            # Sort customers by demand descending to pack big ones first
            order = np.argsort(-demands)
            
            # To ensure validity, we might need to adjust.
            # Let's try a "closest to LP" greedy.
            # For each customer i, look at x[i, :].
            # Pick j that maximizes x[i,j] such that capacity allows.
            
            # To be safe and fast, we can just use the LP solution to guide a greedy.
            # Or simply round x[i,j] and check validity.
            # If invalid, fallback.
            
            # Let's try rounding x[i,j] directly.
            x_mat = x_flat.reshape(n_cust, n_fac)
            
            # Sort customers by "conflict" or just demand
            # Actually, let's just do a greedy pass using LP values as priority.
            
            for i in order:
                # Get LP values for this customer
                lp_vals = x_mat[i].copy()
                
                # Sort facilities by LP value descending (higher LP value = more likely optimal)
                fac_order = np.argsort(-lp_vals)
                
                assigned = False
                for f in fac_order:
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
                
                if not assigned:
                    # Fallback to cheapest feasible
                    fac_order_cost = np.argsort(assign_cost[i])
                    for f in fac_order_cost:
                        if load[f] + demands[i] <= capacities[f]:
                            result[i] = f
                            load[f] += demands[i]
                            break
                    else:
                        return None # Infeasible?
                        
            return result
        else:
            return None
    except Exception:
        return None

def _local_search(demands, capacities, open_cost, assign_cost, assignment, max_iters=500, deadline=None):
    """
    Perform local search to improve the assignment.
    Moves:
    1. Reassign a customer to a different facility (if capacity allows).
    2. Swap two customers assigned to different facilities.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    current_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
    # Add open cost
    open_facilities = set(assignment.tolist())
    current_cost += np.sum([open_cost[f] for f in open_facilities])
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
        
    best_cost = current_cost
    best_assignment = assignment.copy()
    
    # Precompute costs
    # For each customer i and facility j, we have assign_cost[i,j]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try reassigning each customer
        for i in range(n_cust):
            current_fac = assignment[i]
            # Try all other facilities
            for new_fac in range(n_fac):
                if new_fac == current_fac:
                    continue
                
                # Check capacity
                if load[new_fac] + demands[i] <= capacities[new_fac]:
                    # Calculate cost delta
                    old_assign_cost = assign_cost[i, current_fac]
                    new_assign_cost = assign_cost[i, new_fac]
                    
                    old_open_cost = 0.0
                    new_open_cost = 0.0
                    
                    # If current_fac becomes empty, we save open_cost
                    if load[current_fac] == demands[i]:
                        old_open_cost = open_cost[current_fac]
                        
                    # If new_fac was empty, we pay open_cost
                    if load[new_fac] == 0:
                        new_open_cost = open_cost[new_fac]
                        
                    delta = (new_assign_cost + new_open_cost) - (old_assign_cost + old_open_cost)
                    
                    if delta < -1e-9:
                        # Apply move
                        assignment[i] = new_fac
                        load[current_fac] -= demands[i]
                        load[new_fac] += demands[i]
                        current_cost += delta
                        improved = True
                        break # Move to next customer
                        
        # 2. Try swapping two customers
        # This is O(N^2 * F) which might be slow. 
        # We can limit this or do it less frequently.
        if not improved and it % 5 == 0:
            for i in range(n_cust):
                for k in range(i + 1, n_cust):
                    fac_i = assignment[i]
                    fac_k = assignment[k]
                    if fac_i == fac_k:
                        continue
                    
                    # Check capacity for swap
                    # i moves to fac_k, k moves to fac_i
                    # load[fac_k] - demands[k] + demands[i] <= capacities[fac_k]
                    # load[fac_i] - demands[i] + demands[k] <= capacities[fac_i]
                    
                    if load[fac_k] - demands[k] + demands[i] > capacities[fac_k]:
                        continue
                    if load[fac_i] - demands[i] + demands[k] > capacities[fac_i]:
                        continue
                        
                    # Cost delta
                    old_cost = assign_cost[i, fac_i] + assign_cost[k, fac_k]
                    new_cost = assign_cost[i, fac_k] + assign_cost[k, fac_i]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If fac_i becomes empty
                    if load[fac_i] == demands[i]:
                        # fac_i loses i. Does it gain k?
                        # If load[fac_i] - demands[i] + demands[k] == 0, it becomes empty?
                        # No, it gains k. So it stays open unless k is 0 demand (impossible).
                        # Wait, if load[fac_i] == demands[i], then after removing i, load is 0.
                        # Then adding k makes load = demands[k].
                        # So fac_i was open, stays open (unless demands[k]=0).
                        # So no open cost change for fac_i?
                        # Actually, if load[fac_i] == demands[i], it was open.
                        # After swap, load[fac_i] = demands[k]. If demands[k]>0, it's open.
                        # So open cost for fac_i is unchanged.
                        
                        # What if fac_k becomes empty?
                        if load[fac_k] == demands[k]:
                             # fac_k loses k, gains i.
                             # So fac_k stays open.
                             pass
                             
                    # Actually, open cost only changes if a facility becomes empty or non-empty.
                    # In a swap between two distinct facilities:
                    # fac_i: load -> load - d_i + d_k. If load == d_i, it was open. Now load = d_k > 0, so open. No change.
                    # fac_k: load -> load - d_k + d_i. If load == d_k, it was open. Now load = d_i > 0, so open. No change.
                    # So open cost never changes in a swap between two occupied facilities?
                    # Yes, because both remain occupied.
                    
                    delta = new_cost - old_cost
                    if delta < -1e-9:
                        # Apply swap
                        assignment[i] = fac_k
                        assignment[k] = fac_i
                        load[fac_i] -= demands[i]
                        load[fac_i] += demands[k]
                        load[fac_k] -= demands[k]
                        load[fac_k] += demands[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            break
            
    return best_assignment, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Try LP relaxation
    result = _solve_lp(demands, capacities, open_cost, assign_cost)
    
    if result is None:
        # Fallback: Greedy (Largest demand first, cheapest feasible facility)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                result[i] = 0
                
    # 2. Local Search Polish
    if time.time() < deadline - 2.0:
        result, _ = _local_search(demands, capacities, open_cost, assign_cost, result, max_iters=1000, deadline=deadline - 2.0)
        
    return result
# EVOLVE-BLOCK-END
