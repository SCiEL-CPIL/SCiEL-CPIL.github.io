"""Program using Lagrangian Relaxation and Subgradient Optimization for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation with Subgradient Optimization.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res
    
    def local_search(res, load, max_iter=5000):
        """Local search with relocate moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
    
    def lagrangian_subgradient(max_iters=3000):
        """
        Lagrangian Relaxation with Subgradient Optimization.
        Relaxes capacity constraints with multipliers lambda[j].
        For fixed lambda, each customer independently chooses facility minimizing
        assign_cost[i,j] + lambda[j]*demands[i].
        """
        lam = np.zeros(n_fac)
        best_lam = lam.copy()
        best_dual_val = np.inf
        best_primal_res = None
        best_primal_score = np.inf
        
        step_scale = 1.0
        patience = 0
        max_patience = 50
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            eff_cost = assign_cost + np.outer(demands, lam)
            primal_res = np.argmin(eff_cost, axis=1)
            
            load = np.zeros(n_fac)
            np.add.at(load, primal_res, demands)
            
            min_costs = np.min(eff_cost, axis=1)
            dual_val = np.sum(min_costs) - np.dot(lam, capacities)
            
            if dual_val < best_dual_val:
                best_dual_val = dual_val
                best_lam = lam.copy()
            
            if np.all(load <= capacities + 1e-9):
                score = compute_score(primal_res)
                if score < best_primal_score:
                    best_primal_score = score
                    best_primal_res = primal_res.copy()
            
            subgrad = load - capacities
            
            if np.all(np.abs(subgrad) < 1e-6):
                break
            
            ub = best_primal_score if best_primal_res is not None else dual_val + 1.0
            lb = best_dual_val
            gap = max(ub - lb, 1e-9)
            
            norm_sq = np.sum(subgrad ** 2)
            if norm_sq < 1e-12:
                break
            
            alpha = step_scale * gap / norm_sq
            alpha = min(alpha, 1.0)
            
            lam_new = lam + alpha * subgrad
            lam_new = np.maximum(lam_new, 0.0)
            
            if dual_val < best_dual_val - 1e-9:
                patience = 0
                step_scale = min(step_scale * 1.05, 5.0)
            else:
                patience += 1
                if patience > max_patience:
                    step_scale *= 0.7
                    patience = 0
            
            lam = lam_new
        
        return best_primal_res, best_primal_score, best_lam
    
    def repair(primal_res):
        """Repair an infeasible assignment to make it capacity-feasible."""
        res = primal_res.copy()
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        overloaded = load > capacities + 1e-9
        if not np.any(overloaded):
            return res
        
        order = np.argsort(demands)
        for i in order:
            f = res[i]
            if load[f] > capacities[f] + 1e-9:
                best_f = -1
                best_cost = np.inf
                for f_new in range(n_fac):
                    if load[f_new] + demands[i] <= capacities[f_new] + 1e-9:
                        cost = assign_cost[i, f_new]
                        if load[f_new] == 0:
                            cost += open_cost[f_new]
                        if cost < best_cost:
                            best_cost = cost
                            best_f = f_new
                if best_f != -1:
                    load[f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                else:
                    space = capacities - load
                    best_f = np.argmax(space)
                    if space[best_f] >= demands[i]:
                        load[f] -= demands[i]
                        load[best_f] += demands[i]
                        res[i] = best_f
        
        return res
    
    # Get initial feasible solution
    res_greedy = greedy_init()
    load_greedy = np.zeros(n_fac)
    np.add.at(load_greedy, res_greedy, demands)
    local_search(res_greedy, load_greedy, max_iter=2000)
    best_assign = res_greedy.copy()
    best_score = compute_score(best_assign)
    
    # Run Lagrangian Relaxation
    lag_res, lag_score, best_lam = lagrangian_subgradient(max_iters=3000)
    
    if lag_res is not None:
        if lag_score < np.inf:
            if lag_score < best_score:
                best_score = lag_score
                best_assign = lag_res.copy()
        else:
            eff_cost = assign_cost + np.outer(demands, best_lam)
            primal_res = np.argmin(eff_cost, axis=1)
            repaired = repair(primal_res)
            score = compute_score(repaired)
            if score < best_score:
                best_score = score
                best_assign = repaired.copy()
    
    # Run local search on the best solution
    load_best = np.zeros(n_fac)
    np.add.at(load_best, best_assign, demands)
    local_search(best_assign, load_best, max_iter=5000)
    
    # Final verification
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        best_assign = res_greedy
    
    return best_assign.astype(int)

# EVOLVE-BLOCK-END
