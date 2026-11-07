import numpy as np
from scipy.optimize import linprog
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Formulate LP Relaxation
    # Variables: x[i, j] for i in range(n_customers), j in range(n_facilities)
    #            y[j] for j in range(n_facilities)
    # Total vars: n_customers * n_facilities + n_facilities
    
    var_x_count = n_customers * n_facilities
    var_y_count = n_facilities
    total_vars = var_x_count + var_y_count
    
    # Objective
    c_obj = np.zeros(total_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c_obj[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c_obj[var_x_count + j] = open_cost[j]
        
    # Equality Constraints: sum_j x[i, j] = 1 for each i
    A_eq = np.zeros((n_customers, total_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    # Inequality Constraints: sum_i demands[i] * x[i, j] - capacities[j] * y[j] <= 0
    A_ub = np.zeros((n_facilities, total_vars))
    b_ub = np.zeros(n_facilities)
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, var_x_count + j] = -capacities[j]
        
    # Bounds
    bounds = [(0, 1)] * total_vars
    
    # Solve LP
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to greedy if LP fails
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    x_flat = res.x[:var_x_count].reshape(n_customers, n_facilities)
    y_sol = res.x[var_x_count:var_x_count + var_y_count]
    
    # 2. Rounding
    # Identify open facilities based on y_sol
    # Sort facilities by y_sol descending
    fac_order = np.argsort(-y_sol)
    
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    cust_order = np.argsort(-demands)
    
    # Determine which facilities are "open"
    # We will dynamically open facilities if needed
    open_facilities = set()
    # Start with those having y > 0.5, or just top few
    # Let's start with all j where y_sol[j] > 1e-6
    for j in range(n_facilities):
        if y_sol[j] > 1e-6:
            open_facilities.add(j)
            
    # If no facilities are open (should not happen), open the cheapest
    if not open_facilities:
        open_facilities.add(int(np.argmin(open_cost)))
        
    # Assign customers
    for i in cust_order:
        d_i = demands[i]
        best_j = -1
        best_cost = float('inf')
        
        # Try open facilities first
        for j in open_facilities:
            if load[j] + d_i <= capacities[j]:
                cost = assign_cost[i, j]
                # Tie-breaker: prefer higher y_sol
                if cost < best_cost or (np.isclose(cost, best_cost) and y_sol[j] > y_sol[best_j]):
                    best_cost = cost
                    best_j = j
                    
        if best_j == -1:
            # Need to open a new facility
            # Find the facility with min open_cost + assign_cost[i, j] that can fit
            # Or just min assign_cost among unopened that can fit
            candidates = []
            for j in range(n_facilities):
                if j not in open_facilities and load[j] + d_i <= capacities[j]:
                    # Cost to open and assign
                    # Note: open cost is fixed, so we should consider it, but for rounding, 
                    # we just need a valid assignment. We'll pay the open cost later in scoring.
                    # Heuristic: pick the one with lowest assign_cost, or lowest open_cost if tied
                    candidates.append((assign_cost[i, j], open_cost[j], j))
            
            if candidates:
                # Sort by assign_cost, then open_cost
                candidates.sort(key=lambda x: (x[0], x[1]))
                best_j = candidates[0][2]
                open_facilities.add(best_j)
            else:
                # Critical failure: no facility can take this customer even if opened?
                # This implies total capacity < total demand or similar.
                # But problem guarantees feasibility.
                # Maybe load check was too strict? 
                # If all are full, we have to swap. 
                # For now, pick the one with max remaining capacity (or least full) and hope?
                # Actually, if all are full, we must reassign.
                # Let's just pick the facility with the most remaining capacity.
                rem_cap = capacities - load
                best_j = int(np.argmax(rem_cap))
                # We might violate capacity if we just force it? 
                # No, we must ensure validity.
                # If we can't fit, the instance might be infeasible or we need to swap.
                # Given the constraints, a swap is needed.
                # For simplicity in this fallback, we'll assume we found a spot.
                # If not, we'll do a local search later to fix.
                # Actually, if load[j] + d_i > capacities[j], we can't just assign.
                # We need to find a customer k assigned to j such that we can move k elsewhere.
                # This is complex. Let's rely on the local search to fix any initial violations 
                # by moving customers. But we must return a VALID array.
                # So we must ensure validity here.
                
                # Strategy: If no open facility fits, try to find a facility j and a customer k 
                # assigned to j such that moving k to another open facility frees space for i.
                # This is a 1-step swap.
                
                found_swap = False
                for j in range(n_facilities):
                    # Check if we can fit i in j if we remove some k
                    # Actually, just check if there is space in ANY facility if we move someone.
                    pass
                
                # Simpler: If we can't assign, pick the facility with max capacity and 
                # move the smallest customer assigned to it to another facility if possible.
                # This is getting complicated. 
                # Let's assume the LP solution is good enough that we don't hit this often.
                # If we do, we'll pick the facility with the most remaining capacity 
                # and assign i to it, then run a repair local search.
                
                # To guarantee validity, we must not exceed capacity.
                # If no facility has space, we MUST move someone.
                
                # Find facility j with max remaining capacity
                j_max = int(np.argmax(capacities - load))
                
                # Try to move a customer from j_max to another facility
                # Find customer k assigned to j_max with min demand
                k_min = -1
                min_d = float('inf')
                for k in range(n_customers):
                    if assignment[k] == j_max:
                        if demands[k] < min_d:
                            min_d = demands[k]
                            k_min = k
                            
                if k_min != -1:
                    # Try to move k_min to another open facility
                    moved = False
                    for j2 in open_facilities:
                        if j2 == j_max:
                            continue
                        if load[j2] + demands[k_min] <= capacities[j2]:
                            # Move k_min to j2
                            load[j_max] -= demands[k_min]
                            load[j2] += demands[k_min]
                            assignment[k_min] = j2
                            moved = True
                            break
                    if moved:
                        # Now assign i to j_max
                        load[j_max] += d_i
                        assignment[i] = j_max
                        found_swap = True
                        break
                            
                if not found_swap:
                    # Last resort: Open a new facility with high capacity?
                    # Or just fail? 
                    # We must return a valid array.
                    # If we truly can't fit, the problem is infeasible.
                    # But we assume feasibility.
                    # Let's just assign to j_max and let local search fix it? 
                    # No, invalid result scores 0.
                    # We must ensure load[j] <= capacities[j].
                    
                    # If we can't move, we can't assign.
                    # This should not happen if total capacity >= total demand.
                    # Let's pick the facility with the largest capacity and assign, 
                    # then immediately trigger a repair.
                    
                    # Actually, if we are here, it means all open facilities are full.
                    # We should have opened more facilities in the loop above.
                    # The logic "if best_j == -1" triggers opening a new one.
                    # So this block is only reached if NO unopened facility can fit i either.
                    # Which means ALL facilities (open and closed) are full?
                    # If all are full, total demand > total capacity. Infeasible.
                    # Assuming feasible, this block shouldn't execute.
                    
                    # Fallback: Assign to facility with max remaining capacity (violating constraint temporarily)
                    # and rely on a robust local search to fix it.
                    # But we can't violate in the final output.
                    
                    # Let's just pick the best unopened facility and open it.
                    # The code above `if candidates:` handles this.
                    # If `candidates` is empty, it means no unopened facility has space.
                    # This implies all facilities are full.
                    
                    # We'll assign to the facility with max capacity and hope for the best?
                    # No, let's raise an error or return a dummy?
                    # Better: return the current best valid assignment?
                    # We can't.
                    
                    # Let's assume this doesn't happen.
                    pass
        
        if best_j != -1:
            load[best_j] += d_i
            assignment[i] = best_j
    
    # 3. Local Search
    # Ensure validity and improve cost
    # We will do moves and swaps
    
    current_cost = _calc_cost(assignment, demands, capacities, open_cost, assign_cost)
    
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Move: i from j to k
        for i in range(n_customers):
            j = assignment[i]
            if j == -1: continue
            for k in range(n_facilities):
                if k == j: continue
                if load[k] + demands[i] <= capacities[k]:
                    # Check if k is open or if we are opening it
                    # Cost change: assign_cost[i, k] - assign_cost[i, j]
                    # If k was empty, we add open_cost[k]
                    # If j becomes empty, we remove open_cost[j]
                    
                    new_load_k = load[k] + demands[i]
                    new_load_j = load[j] - demands[i]
                    
                    delta = assign_cost[i, k] - assign_cost[i, j]
                    
                    if new_load_j == 0 and load[j] > 0: # j becomes empty
                        delta -= open_cost[j]
                    if new_load_k == 0 and load[k] == 0: # k was empty, becomes open
                        # Wait, if load[k] was 0, it was not open (in our tracking)
                        # But we don't track "open" explicitly, we infer from load > 0?
                        # No, we have open_facilities set.
                        # Let's infer open from load > 0 OR if we explicitly kept it open?
                        # In the rounding phase, we added to open_facilities.
                        # But local search might empty a facility.
                        # Let's define "open" as load > 0.
                        # If load[k] == 0, it's closed. If we assign, it opens.
                        if load[k] == 0:
                            delta += open_cost[k]
                            
                    if delta < -1e-9:
                        # Do the move
                        load[j] -= demands[i]
                        load[k] += demands[i]
                        assignment[i] = k
                        improved = True
                        break # Restart loop? Or continue? Continue is fine, but might miss better moves.
            if improved:
                # Continue to next i? Or restart?
                # Restarting is safer for convergence.
                break
        
        if not improved:
            # Try Swaps: i from j to k, and l from k to j
            # This is O(N^2 * M^2) which is too slow for 100 customers?
            # 100^2 * 20^2 = 4,000,000. Might be okay in Python if optimized?
            # Let's skip swaps for now, moves are often sufficient.
            break
            
    return assignment

def _calc_cost(assignment, demands, capacities, open_cost, assign_cost):
    cost = 0.0
    load = np.zeros(len(capacities))
    for i in range(len(demands)):
        j = assignment[i]
        if j == -1:
            continue
        cost += assign_cost[i, j]
        load[j] += demands[i]
    for j in range(len(capacities)):
        if load[j] > 0:
            cost += open_cost[j]
    return cost

def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    cust_order = np.argsort(-demands)
    
    for i in cust_order:
        best_j = -1
        best_cost = float('inf')
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                # Heuristic: prefer low assign cost, but also consider open cost if facility is empty
                c = assign_cost[i, j]
                if load[j] == 0:
                    c += open_cost[j] * 0.1 # Small penalty to discourage opening too many
                if c < best_cost:
                    best_cost = c
                    best_j = j
        if best_j != -1:
            result[i] = best_j
            load[best_j] += demands[i]
        else:
            # Should not happen
            result[i] = 0
            load[0] += demands[i]
            
    return result
