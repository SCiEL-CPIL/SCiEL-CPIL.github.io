"""Program using Lagrangian Relaxation and Subgradient Optimization for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation.
    
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
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                # If no facility has space, assign to the one with most remaining space
                # (This should rarely happen if total capacity >= total demand)
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    # Fallback: assign to cheapest facility regardless (will be infeasible)
                    res[i] = fac_order[0]
        return res
    
    def local_search(res, load, max_iter=10000):
        """
        Local search with relocate moves.
        Tries to improve by moving one customer at a time.
        """
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            # Try relocate moves for each customer
            # Use random order to avoid bias
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                # Find best alternative
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    # Open cost change
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                
                if best_f != -1:
                    # Apply move
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
    
    def compute_lambda_value(lam):
        """
        Compute L(lam) and the corresponding assignment (which may violate capacities).
        Returns: (value, assignment, used_facilities)
        """
        c_eff = assign_cost - np.outer(demands, lam)
        j_i = np.argmin(c_eff, axis=1)
        
        used_fac = np.zeros(n_fac, dtype=bool)
        used_fac[j_i] = True
        
        term1 = np.dot(lam, capacities)
        term2 = np.sum(open_cost[used_fac])
        term3 = np.sum(c_eff[np.arange(n_cust), j_i])
        
        value = term1 + term2 + term3
        return value, j_i, used_fac
    
    def round_to_feasible(j_i):
        """
        Convert a potentially infeasible assignment to a feasible one.
        """
        res = j_i.copy()
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        for _ in range(200):
            overloaded = np.where(load > capacities + 1e-9)[0]
            if len(overloaded) == 0:
                break
            
            # Pick the most overloaded facility
            f_over = overloaded[np.argmax(load[overloaded] - capacities[overloaded])]
            
            # Find a customer in f_over to move to a facility with space
            custs_in_f = np.where(res == f_over)[0]
            best_cust = -1
            best_alt = -1
            best_cost_inc = np.inf
            
            for i in custs_in_f:
                space = capacities - load
                space[f_over] = -1
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
                break
            
            res[best_cust] = best_alt
            load[f_over] -= demands[best_cust]
            load[best_alt] += demands[best_cust]
        
        return res
    
    # Initial greedy solution
    res = greedy_init()
    load = np.zeros(n_fac)
    np.add.at(load, res, demands)
    
    # Run local search on initial solution
    local_search(res, load)
    best_score = compute_score(res)
    best_assign = res.copy()
    
    # Lagrangian Relaxation
    lam = np.zeros(n_fac)
    max_iter = 3000
    step_scale = 1.0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        val, j_i, used_fac = compute_lambda_value(lam)
        
        # Get feasible solution from rounding
        feasible_res = round_to_feasible(j_i)
        load_check = np.zeros(n_fac)
        np.add.at(load_check, feasible_res, demands)
        
        if not np.any(load_check > capacities + 1e-9):
            # Run local search on the rounded solution
            local_search(feasible_res, load_check, max_iter=500)
            score = compute_score(feasible_res)
            
            if score < best_score:
                best_score = score
                best_assign = feasible_res.copy()
        
        # Compute subgradient
        load_lam = np.zeros(n_fac)
        np.add.at(load_lam, j_i, demands)
        grad = capacities - load_lam
        
        # Check convergence
        if np.allclose(grad, 0, atol=1e-8):
            break
        
        # Update lambda
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-10:
            break
        
        # Step size: (UB - L_k) / ||grad||^2
        step = step_scale * (best_score - val) / grad_norm_sq
        if step < 0:
            step = 0.01
        
        lam = lam + step * grad
        lam = np.maximum(lam, 0)
        
        # Decay step scale
        if it % 500 == 0 and it > 0:
            step_scale *= 0.95
    
    # Final local search on best solution found
    load_final = np.zeros(n_fac)
    np.add.at(load_final, best_assign, demands)
    local_search(best_assign, load_final, max_iter=5000)
    
    # Verify feasibility
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        # Fallback to greedy if somehow infeasible
        best_assign = greedy_init()
    
    return best_assign.astype(int)

# EVOLVE-BLOCK-END
