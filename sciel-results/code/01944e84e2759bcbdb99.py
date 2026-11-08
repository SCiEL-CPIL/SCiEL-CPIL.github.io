"""Program using Lagrangian Relaxation with Subgradient Optimization and Enhanced Local Search for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation and Local Search.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices for each customer to speed up repair/local search if needed
    # Though vectorized argmin is fast, having sorted lists helps in specific repair heuristics
    sorted_fac_indices = np.argsort(assign_cost, axis=1)
    
    def calc_cost(assignment):
        """Calculate total cost and check feasibility. Returns (cost, is_feasible, load)."""
        load = np.zeros(n_fac)
        np.add.at(load, assignment, demands)
        
        if np.any(load > capacities + 1e-9):
            return np.inf, False, load
            
        open_fac = load > 0
        cost = np.sum(open_cost[open_fac]) + np.sum(assign_cost[np.arange(n_cust), assignment])
        return cost, True, load

    def greedy_init():
        """Greedy initialization: sort by demand desc, assign to cheapest feasible facility."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        
        for i in order:
            # Get facilities sorted by cost for this customer
            fac_order = sorted_fac_indices[i]
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: assign to facility with most remaining capacity (or first if all full)
                # This shouldn't happen if total capacity >= total demand, which is standard
                # But for safety, assign to the one with most space
                space = capacities - load
                f_best = np.argmax(space)
                res[i] = f_best
                load[f_best] += demands[i]
                
        return res

    best_assign = greedy_init()
    best_cost, _, best_load = calc_cost(best_assign)
    
    # If greedy failed (infeasible), try a simple repair
    if not np.isfinite(best_cost):
         # Simple repair: move customers from overloaded to underloaded
         # This is a fallback, greedy should usually work
         pass 

    # Lagrangian Relaxation Setup
    # Minimize sum_j open_cost[j] y_j + sum_i sum_j assign_cost[i,j] x_ij
    # s.t. sum_j x_ij = 1
    #      sum_i demands[i] x_ij <= capacities[j]
    #
    # Relax capacity: L(lambda) = sum_j lambda[j] capacities[j] + sum_i min_j (assign_cost[i,j] - lambda[j] demands[i])
    # Note: The open cost is handled in the primal evaluation. The Lagrangian dual provides a lower bound and a heuristic assignment.
    
    lam = np.zeros(n_fac)
    
    # Subgradient parameters
    max_iter = 5000
    step_scale = 1.0
    best_ub = best_cost
    
    # Track best solution found during Lagrangian iterations
    lag_best_assign = best_assign.copy()
    lag_best_cost = best_cost
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0: # Leave time for final local search
            break
            
        # 1. Solve Lagrangian Subproblem
        # Effective cost for customer i to facility j: assign_cost[i,j] - lam[j] * demands[i]
        c_eff = assign_cost - np.outer(demands, lam)
        
        # Assign each customer to the facility with minimum effective cost
        j_i = np.argmin(c_eff, axis=1)
        
        # 2. Calculate Lagrangian Lower Bound (LB)
        # LB = sum_j lam[j] capacities[j] + sum_i c_eff[i, j_i]
        # Note: This LB does not include open costs directly, but the primal solution derived from j_i
        # will be evaluated with open costs to get a UB.
        
        load_lag = np.zeros(n_fac)
        np.add.at(load_lag, j_i, demands)
        
        # 3. Check Feasibility of Lagrangian Solution
        if np.all(load_lag <= capacities + 1e-9):
            cost_lag, _, _ = calc_cost(j_i)
            if cost_lag < lag_best_cost:
                lag_best_cost = cost_lag
                lag_best_assign = j_i.copy()
                best_ub = cost_lag
        else:
            # 4. Repair Infeasible Solution to get a UB
            # Simple repair: While any facility is overloaded, move a customer to the best feasible alternative
            res_repair = j_i.copy()
            load_repair = load_lag.copy()
            
            # Limit repair steps to avoid slow loops
            for _ in range(50):
                overloaded = np.where(load_repair > capacities + 1e-9)[0]
                if len(overloaded) == 0:
                    break
                
                # Pick the most overloaded facility
                f_over = overloaded[np.argmax(load_repair[overloaded] - capacities[overloaded])]
                
                # Find a customer in f_over to move
                custs_in_f = np.where(res_repair == f_over)[0]
                best_cust = -1
                best_alt = -1
                best_cost_inc = np.inf
                
                # Vectorized search for best move
                # For each customer in f_over, find the best facility with space
                # This is a bit complex to vectorize fully without loops due to dynamic 'space'
                # So we iterate customers in f_over, but vectorize the facility check
                
                for i in custs_in_f:
                    space = capacities - load_repair
                    space[f_over] = -1 # Cannot stay in same facility
                    valid_f = np.where(space >= demands[i])[0]
                    
                    if len(valid_f) == 0:
                        continue
                        
                    costs = assign_cost[i, valid_f]
                    min_idx = np.argmin(costs)
                    f_alt = valid_f[min_idx]
                    cost_inc = assign_cost[i, f_alt] - assign_cost[i, f_over]
                    
                    if cost_inc < best_cost_inc:
                        best_cost_inc = cost_inc
                        best_cust = i
                        best_alt = f_alt
                        
                if best_cust == -1:
                    # Cannot repair with single moves? Try swapping?
                    # For simplicity in this loop, break. Local search will handle complex moves later.
                    break
                    
                # Apply move
                res_repair[best_cust] = best_alt
                load_repair[f_over] -= demands[best_cust]
                load_repair[best_alt] += demands[best_cust]
                
            # Check if repaired solution is feasible
            if np.all(load_repair <= capacities + 1e-9):
                cost_repair, _, _ = calc_cost(res_repair)
                if cost_repair < lag_best_cost:
                    lag_best_cost = cost_repair
                    lag_best_assign = res_repair.copy()
                    best_ub = cost_repair

        # 5. Update Lagrange Multipliers (Subgradient Method)
        # Gradient of LB w.r.t lambda_j is: capacities[j] - load_lag[j]
        grad = capacities - load_lag
        
        # Check convergence
        if np.allclose(grad, 0, atol=1e-6):
            break
            
        # Step size
        # step = (UB - LB) / ||grad||^2
        # We need LB. 
        # LB = sum_j lam[j] capacities[j] + sum_i c_eff[i, j_i]
        lb_val = np.dot(lam, capacities) + np.sum(c_eff[np.arange(n_cust), j_i])
        
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-10:
            break
            
        # Ensure UB is valid
        if not np.isfinite(best_ub):
             best_ub = lag_best_cost if np.isfinite(lag_best_cost) else 1e10
             
        step = step_scale * (best_ub - lb_val) / grad_norm_sq
        
        # If step is negative (LB > UB, which can happen due to non-convexity or numerical issues), reduce step
        if step < 0:
            step = 0.01
            
        # Update lambda, ensure non-negative (since constraints are <=)
        lam = np.maximum(0, lam + step * grad)
        
        # Optional: Damping to prevent oscillation
        # lam = 0.5 * lam + 0.5 * lam_prev (not implemented here for simplicity, standard step usually works)

    # Final Local Search on the best solution found
    # Use the best of the initial greedy or the Lagrangian-derived solutions
    final_assign = lag_best_assign.copy()
    final_load, _, _ = calc_cost(final_assign)
    
    # If Lagrangian didn't improve, stick with greedy
    if lag_best_cost >= best_cost:
        final_assign = best_assign.copy()
        final_load = best_load.copy()
    
    # Run Local Search
    # We use a time-limited local search with relocation and swap moves.
    
    def local_search(res, load):
        improved = True
        # Precompute costs for all i, f to avoid repeated indexing? 
        # assign_cost is already available.
        
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Relocation Moves
            # For each customer, check if moving to another facility reduces cost
            # Delta cost = assign_cost[i, f_new] - assign_cost[i, f_old]
            #             + open_cost_change
            # open_cost_change:
            #   if load[f_new] == 0: +open_cost[f_new]
            #   if load[f_old] == demands[i]: -open_cost[f_old]
            
            for i in range(n_cust):
                current_f = res[i]
                d_i = demands[i]
                
                # Check all other facilities
                # We can vectorize the cost calculation for all f_new
                # But we need to check capacity feasibility: load[f_new] + d_i <= capacities[f_new]
                
                # Find feasible facilities
                space = capacities - load
                feasible_f = np.where(space >= d_i)[0]
                
                if len(feasible_f) == 0:
                    continue
                    
                # Exclude current facility from consideration for "improvement" via move
                # (Though moving to same facility is a no-op)
                feasible_f = feasible_f[feasible_f != current_f]
                
                if len(feasible_f) == 0:
                    continue
                
                # Calculate cost delta for each feasible f
                # Base delta: assign_cost[i, f] - assign_cost[i, current_f]
                base_delta = assign_cost[i, feasible_f] - assign_cost[i, current_f]
                
                # Open cost adjustments
                # If f is empty, add open_cost[f]
                empty_mask = load[feasible_f] == 0
                base_delta[empty_mask] += open_cost[feasible_f[empty_mask]]
                
                # If current_f becomes empty, subtract open_cost[current_f]
                # This term is constant for all f in this iteration, so we can add it to the minimum later
                # Or just add it to all deltas? No, it only applies if we actually move.
                # Let's compute total delta = base_delta + (open_cost[current_f] if load[current_f] == d_i else 0)
                # Wait, if we move i from current_f to f_new:
                # Cost change = (assign_cost[i, f_new] - assign_cost[i, current_f]) 
                #              + (open_cost[f_new] if load[f_new]==0 else 0)
                #              - (open_cost[current_f] if load[current_f]==d_i else 0)
                
                current_open_penalty = open_cost[current_f] if load[current_f] == d_i else 0.0
                
                total_deltas = base_delta - current_open_penalty
                
                min_idx = np.argmin(total_deltas)
                min_delta = total_deltas[min_idx]
                
                if min_delta < -1e-9:
                    f_new = feasible_f[min_idx]
                    # Apply move
                    load[current_f] -= d_i
                    load[f_new] += d_i
                    res[i] = f_new
                    improved = True
                    break # Restart loop after a move to update loads
                    
            if improved:
                continue
            
            # 2. Swap Moves
            # If no relocation improves, try swaps.
            # Swap i and k if res[i] != res[k]
            # Conditions:
            #   load[res[i]] - demands[i] + demands[k] <= capacities[res[i]]
            #   load[res[k]] - demands[k] + demands[i] <= capacities[res[k]]
            # Cost Delta:
            #   (assign_cost[i, res[k]] - assign_cost[i, res[i]]) + (assign_cost[k, res[i]] - assign_cost[k, res[k]])
            #   + Open cost changes for res[i] and res[k]
            
            # To keep it efficient, we only check a subset of pairs or use a heuristic.
            # Given 100 customers, O(N^2) is 10,000 pairs. This is fast.
            
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_i = res[i]
                d_i = demands[i]
                
                for k in range(i+1, n_cust):
                    f_k = res[k]
                    if f_i == f_k:
                        continue
                    d_k = demands[k]
                    
                    # Check capacity for swap
                    # Facility f_i loses d_i, gains d_k
                    new_load_i = load[f_i] - d_i + d_k
                    if new_load_i > capacities[f_i] + 1e-9:
                        continue
                        
                    # Facility f_k loses d_k, gains d_i
                    new_load_k = load[f_k] - d_k + d_i
                    if new_load_k > capacities[f_k] + 1e-9:
                        continue
                    
                    # Calculate cost delta
                    assign_delta = (assign_cost[i, f_k] - assign_cost[i, f_i]) + \
                                   (assign_cost[k, f_i] - assign_cost[k, f_k])
                    
                    open_delta = 0.0
                    # If f_i becomes empty (new_load_i == 0), we save open_cost[f_i]
                    if new_load_i == 0:
                        open_delta -= open_cost[f_i]
                    # If f_k becomes empty (new_load_k == 0), we save open_cost[f_k]
                    if new_load_k == 0:
                        open_delta -= open_cost[f_k]
                        
                    # Note: If a facility was NOT empty before and NOT empty after, no change.
                    # If it was empty before? Impossible, because i is assigned to it.
                    # So we only subtract open cost if it becomes empty.
                    
                    total_delta = assign_delta + open_delta
                    
                    if total_delta < -1e-9:
                        # Apply swap
                        load[f_i] = new_load_i
                        load[f_k] = new_load_k
                        res[i] = f_k
                        res[k] = f_i
                        improved = True
                        break # Restart outer loop
                if improved:
                    break
                    
    local_search(final_assign, final_load)
    
    # Final verification
    final_load_check = np.zeros(n_fac)
    np.add.at(final_load_check, final_assign, demands)
    if np.any(final_load_check > capacities + 1e-9):
        # Fallback to greedy if local search broke feasibility (should not happen)
        return best_assign
        
    return final_assign.astype(int)

# EVOLVE-BLOCK-END
