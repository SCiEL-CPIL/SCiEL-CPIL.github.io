"""Capacitated facility location: Lagrangian Relaxation with Subgradient Optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation with 
    Subgradient Optimization, followed by a final feasibility repair.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    deadline = time.time() + 140.0
    
    # 1. Generate a valid initial solution using a simple greedy heuristic
    # Sort customers by demand descending (largest first)
    cust_order = np.argsort(-demands)
    init_assign = np.zeros(n_cust, dtype=int)
    current_load = np.zeros(n_fac)
    
    for c in cust_order:
        # Find feasible facilities
        feasible = np.where(current_load + demands[c] <= capacities)[0]
        if len(feasible) == 0:
            # If no feasible facility, pick one with max remaining capacity (will be repaired later)
            f = np.argmax(capacities - current_load)
        else:
            # Minimize assignment cost + marginal open cost penalty
            # Approximate marginal open cost: if facility is empty, add open_cost
            costs = assign_cost[c, feasible]
            # Add open cost if facility is currently empty
            costs += np.where(current_load[feasible] == 0, open_cost[feasible], 0)
            f = feasible[np.argmin(costs)]
        
        init_assign[c] = f
        current_load[f] += demands[c]
        
    # Ensure validity of initial solution (repair overloads if any)
    # Simple repair: move customers from overloaded facilities to least costly feasible ones
    for _ in range(10): # Max 10 repair passes
        overloaded = np.where(current_load > capacities)[0]
        if len(overloaded) == 0:
            break
        for f in overloaded:
            # Find customers in this facility
            custs_in_f = np.where(init_assign == f)[0]
            # Move the one with smallest demand first? Or largest? 
            # Moving largest might free space faster for others, but might be more costly.
            # Let's try moving the one with highest assignment cost to other feasible places.
            # Actually, just move one that fits elsewhere.
            moved = False
            # Sort by demand descending to free up space quickly
            custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
            for c in custs_in_f:
                # Find feasible facility for c
                feasible = np.where(current_load + demands[c] <= capacities)[0]
                feasible = feasible[feasible != f] # Don't move to same
                if len(feasible) > 0:
                    costs = assign_cost[c, feasible]
                    # Add open cost if facility is empty
                    costs += np.where(current_load[feasible] == 0, open_cost[feasible], 0)
                    new_f = feasible[np.argmin(costs)]
                    init_assign[c] = new_f
                    current_load[f] -= demands[c]
                    current_load[new_f] += demands[c]
                    moved = True
                    break # Restart loop for this facility if we moved one
            if not moved:
                # Cannot fix, break
                break
        
    best_assign = init_assign.copy()
    best_score = np.sum(open_cost[np.unique(best_assign)]) + np.sum(assign_cost[np.arange(n_cust), best_assign])
    
    # 2. Lagrangian Relaxation
    # Dual variables lambda_f >= 0
    lambda_ = np.zeros(n_fac)
    
    # Subgradient method parameters
    step_size = 1.0
    max_iter = 5000
    
    for t in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # --- Primal Problem (Relaxed) ---
        # Minimize sum_c min_f (assign_cost[c,f] + lambda_f * demand_c) + sum_f open_cost_f * y_f
        # Here y_f is implicitly 1 if any customer is assigned to f, else 0.
        # Since we don't have a direct y_f in the decomposed problem, we just compute the 
        # lower bound and the solution.
        
        # For each customer c, choose f minimizing assign_cost[c,f] + lambda_f * demand_c
        # Note: This relaxation does not explicitly include open_cost in the customer decision,
        # but the dual function is:
        # g(lambda) = sum_c min_f (assign_cost[c,f] + lambda_f * demand_c) + sum_f open_cost_f * y_f(lambda)
        # Wait, the standard Lagrangian relaxation for CFL relaxes capacity constraints:
        # L(y, x, lambda) = sum_f open_cost_f y_f + sum_c sum_f assign_cost[c,f] x_cf
        #                    + sum_f lambda_f (sum_c demand_c x_cf - capacity_f)
        # Rearranging terms for fixed lambda:
        # L = sum_f open_cost_f y_f + sum_f lambda_f (-capacity_f) 
        #   + sum_c [ min_f (assign_cost[c,f] + lambda_f demand_c) * x_cf? No, x_cf is binary ]
        # Actually, the min over x for fixed y is:
        # sum_c min_{f: y_f=1} (assign_cost[c,f] + lambda_f demand_c)
        # This is coupled by y_f. 
        # However, a common approach is to ignore the y_f coupling in the inner minimization 
        # and just compute the bound, or treat it as a facility location subproblem.
        
        # Simpler approach: Use the "Dualizing" approach where we just update lambda 
        # based on the load from the solution of the relaxed problem.
        # The relaxed problem for fixed lambda is:
        # Minimize sum_f open_cost_f y_f + sum_c min_{f} (assign_cost[c,f] + lambda_f demand_c) 
        # subject to: y_f = 1 if any c assigned to f.
        
        # Let's compute the assignment that minimizes the Lagrangian term for each customer,
        # assuming all facilities are "open" for the sake of finding the gradient direction.
        # This is a standard heuristic for CFL:
        # 1. For each c, find f minimizing assign_cost[c,f] + lambda_f * demand_c
        # 2. Compute load_f
        # 3. Update lambda_f based on load_f - capacity_f
        
        # Calculate effective cost
        # eff_cost[c, f] = assign_cost[c, f] + lambda_f * demand_c
        eff_cost = assign_cost + np.outer(demands, lambda_)
        
        # Assign each customer to min cost facility
        curr_assign = np.argmin(eff_cost, axis=1)
        
        # Calculate loads
        curr_load = np.zeros(n_fac)
        np.add.at(curr_load, curr_assign, demands)
        
        # Calculate the lower bound (Dual Function Value)
        # LB = sum_c min_f (assign_cost[c,f] + lambda_f * demand_c) 
        #    + sum_f open_cost_f * y_f 
        #    - sum_f lambda_f * capacity_f
        # Note: The y_f term is tricky. If we assume the solution is valid for open costs,
        # we can estimate the objective of the current solution.
        # For the subgradient update, we need the subgradient of the dual function.
        # The subgradient is (load_f - capacity_f).
        
        # Calculate current solution cost (for tracking best)
        # We need to check if curr_assign is feasible (capacity wise).
        # It likely isn't. But we can repair it to get a primal feasible solution.
        
        # --- Repair to get a feasible primal solution ---
        # This is expensive to do every iteration. 
        # Instead, we can keep the best *primal feasible* solution found so far.
        # Let's do a light repair every few iterations or if we have time.
        
        if t % 10 == 0:
            # Attempt to repair curr_assign
            rep_assign = curr_assign.copy()
            rep_load = curr_load.copy()
            
            # Simple greedy repair for overloads
            for _ in range(5): # Max 5 passes
                overloaded = np.where(rep_load > capacities)[0]
                if len(overloaded) == 0:
                    break
                for f in overloaded:
                    custs_in_f = np.where(rep_assign == f)[0]
                    # Move customer with highest demand to a feasible facility
                    custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
                    moved = False
                    for c in custs_in_f:
                        feasible = np.where(rep_load + demands[c] <= capacities)[0]
                        feasible = feasible[feasible != f]
                        if len(feasible) > 0:
                            # Choose min cost
                            costs = assign_cost[c, feasible]
                            # Add open cost penalty if facility is empty in rep_load
                            # Note: rep_load is the load, not the open status. 
                            # We don't track open status in rep_load easily without extra arrays.
                            # Let's just use assign cost for repair.
                            new_f = feasible[np.argmin(costs)]
                            rep_assign[c] = new_f
                            rep_load[f] -= demands[c]
                            rep_load[new_f] += demands[c]
                            moved = True
                            break
                    if not moved:
                        break
            
            # Check if repair was successful
            if np.all(rep_load <= capacities):
                # Calculate score
                used = np.unique(rep_assign)
                score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), rep_assign])
                if score < best_score:
                    best_score = score
                    best_assign = rep_assign.copy()
                    
        # --- Update Lambda ---
        # Subgradient: g = load - capacity
        # Only positive part? No, lambda must be >= 0.
        # If load > capacity, increase lambda.
        # If load < capacity, decrease lambda (but keep >= 0).
        
        g = curr_load - capacities
        
        # Step size decay
        # step_size = alpha / norm(g)
        norm_g = np.linalg.norm(g)
        if norm_g > 0:
            step_size = 0.5 / norm_g # Heuristic step size
            lambda_ = np.maximum(0, lambda_ + step_size * g)
        else:
            break # Converged?
            
    # 3. Final Polish with Local Search on the best found solution
    # Use the best_assign from Lagrangian as a seed for a quick local search
    # This ensures we exploit the proximity to optimality found by the dual search.
    
    if time.time() < deadline - 10.0:
        # Implement a simple local search
        # 1. Single moves
        # 2. Swap moves
        
        res = best_assign.copy()
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # Ensure validity
        if np.any(load > capacities):
             # Repair again
             for _ in range(20):
                 overloaded = np.where(load > capacities)[0]
                 if len(overloaded) == 0: break
                 for f in overloaded:
                     custs_in_f = np.where(res == f)[0]
                     custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
                     moved = False
                     for c in custs_in_f:
                         feasible = np.where(load + demands[c] <= capacities)[0]
                         feasible = feasible[feasible != f]
                         if len(feasible) > 0:
                             new_f = feasible[np.argmin(assign_cost[c, feasible])]
                             res[c] = new_f
                             load[f] -= demands[c]
                             load[new_f] += demands[c]
                             moved = True
                             break
                     if not moved: break
                 
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Single Move Pass
            for c in range(n_cust):
                if time.time() > deadline - 5.0: break
                f_old = res[c]
                d_c = demands[c]
                
                # Find best new facility
                # Feasible: load + d_c <= capacities
                feasible_mask = (load + d_c <= capacities)
                feasible_mask[f_old] = False # Don't move to same
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate delta cost
                # Delta = (assign_cost[c, f_new] - assign_cost[c, f_old]) 
                #       + (open_cost change)
                
                # Open cost change:
                # If f_old becomes empty, save open_cost[f_old]
                # If f_new was empty, pay open_cost[f_new]
                
                base_assign = assign_cost[c, f_old]
                
                # Vectorized calculation of potential new facilities
                # We need to know which facilities are currently empty (load == 0)
                empty_mask = (load == 0)
                
                # Potential new facilities
                cand_f = np.where(feasible_mask)[0]
                
                if len(cand_f) == 0:
                    continue
                    
                # Cost for each candidate
                costs = assign_cost[c, cand_f]
                
                # Open cost adjustment
                # If f_old becomes empty: -open_cost[f_old]
                # If cand_f is empty: +open_cost[cand_f]
                
                delta_open = np.zeros(len(cand_f))
                
                if load[f_old] == d_c:
                    # f_old will be empty
                    # Subtract open_cost[f_old] from all deltas
                    pass # We'll handle this in the total delta calculation
                
                # Add open cost for candidates that are currently empty
                cand_empty = empty_mask[cand_f]
                delta_open[cand_empty] = open_cost[cand_f[cand_empty]]
                
                total_delta = (costs - base_assign) + delta_open
                
                # If f_old becomes empty, we save open_cost[f_old]
                # This applies to ALL moves from f_old
                if load[f_old] == d_c:
                    total_delta -= open_cost[f_old]
                    
                best_idx = np.argmin(total_delta)
                best_delta = total_delta[best_idx]
                
                if best_delta < -1e-9:
                    f_new = cand_f[best_idx]
                    res[c] = f_new
                    load[f_old] -= d_c
                    load[f_new] += d_c
                    improved = True
                    
            # Swap Move Pass (sampled)
            if time.time() > deadline - 5.0: break
            for _ in range(1000):
                if time.time() > deadline - 5.0: break
                i, j = np.random.randint(0, n_cust, size=2)
                if i == j: continue
                if i > j: i, j = j, i
                
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j: continue
                
                d_i = demands[i]
                d_j = demands[j]
                
                # Check capacity
                if load[f_j] - d_j + d_i > capacities[f_j]: continue
                if load[f_i] - d_i + d_j > capacities[f_i]: continue
                
                # Calculate delta
                delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                               (assign_cost[j, f_i] - assign_cost[j, f_j])
                
                # Open cost delta
                # f_i might become empty? load[f_i] - d_i == 0?
                # f_j might become empty? load[f_j] - d_j == 0?
                # But they are swapping, so loads change.
                
                # Simplified: Only consider assignment cost for swaps, 
                # as open cost changes are rare and complex.
                # Or handle it:
                # If load[f_i] == d_i, f_i becomes empty? No, d_j moves in.
                # If load[f_i] - d_i == 0, then after swap, load is d_j. 
                # If d_j > 0, it's not empty.
                # So open cost only changes if a facility becomes empty OR if a facility was empty and becomes non-empty.
                # Since f_i and f_j are both non-empty before swap (they have customers),
                # they remain non-empty after swap (unless one customer is the only one, but then the other moves in).
                # Wait, if load[f_i] == d_i, and d_j moves in, load becomes d_j.
                # If d_j > 0, it's open.
                # If load[f_i] > d_i, it was open, remains open.
                # So open cost DOES NOT CHANGE in a swap between two non-empty facilities.
                
                if delta_assign < -1e-9:
                    res[i], res[j] = res[j], res[i]
                    load[f_i] -= d_i
                    load[f_i] += d_j
                    load[f_j] -= d_j
                    load[f_j] += d_i
                    improved = True

        # Check final validity and score
        if np.all(load <= capacities):
             final_score = np.sum(open_cost[np.unique(res)]) + np.sum(assign_cost[np.arange(n_cust), res])
             if final_score < best_score:
                 best_assign = res.copy()

    return best_assign
# EVOLVE-BLOCK-END
