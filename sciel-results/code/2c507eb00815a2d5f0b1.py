"""Capacitated facility location: Lagrangian relaxation + subgradient + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian relaxation with
    subgradient optimization, followed by local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def _total_cost(result):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _is_feasible(result, load):
        return np.all(load <= capacities + 1e-6)

    def _greedy_feasible(cost_matrix):
        """Greedy assignment using given cost matrix, ensuring feasibility."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac, dtype=float)
        # Place largest demands first
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(cost_matrix[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: facility with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    # Force assign to facility with least excess
                    excess = load - capacities
                    f = int(np.argmin(excess))
                    load[f] += demands[i]
                    result[i] = f
        return result, load

    def _lagrangian_relax(lam):
        """
        Solve the Lagrangian relaxation: assign each customer to the facility
        minimizing assign_cost[c,f] + lam[f] * demands[c], plus open_cost if facility
        is opened. Returns assignment and dual objective value.
        """
        # Effective cost for each (customer, facility) pair
        eff_cost = assign_cost + np.outer(demands, lam)
        
        # For each customer, pick the facility with minimum effective cost
        best_f = np.argmin(eff_cost, axis=1)
        
        # Determine which facilities are opened
        opened = np.zeros(n_fac, dtype=bool)
        for f in range(n_fac):
            if np.any(best_f == f):
                opened[f] = True
        
        # Compute Lagrangian objective
        assign_sum = np.sum(eff_cost[np.arange(n_cust), best_f])
        open_sum = np.sum(open_cost[opened])
        # Subtract penalty for capacity (dual term)
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[best_f[c]] += demands[c]
        dual_obj = assign_sum + open_sum - np.dot(lam, load)
        
        return best_f, load, dual_obj, opened

    def _subgradient_optimize(max_iters=200, deadline=None):
        """
        Subgradient method to find good Lagrange multipliers.
        Returns best feasible solution found and its cost.
        """
        # Initialize multipliers
        lam = np.zeros(n_fac)
        
        # Initial step size
        # Upper bound on optimal objective: sum of all open costs + max assign cost * n_cust
        ub_open = np.sum(open_cost)
        ub_assign = np.max(assign_cost) * n_cust
        UB = ub_open + ub_assign
        
        LB = -np.inf
        step_scale = 2.0
        best_feasible = None
        best_feasible_cost = float('inf')
        best_lam = lam.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 10.0:
                break
            
            # Solve Lagrangian relaxation
            best_f, load, dual_obj, opened = _lagrangian_relax(lam)
            
            # Update lower bound
            if dual_obj > LB:
                LB = dual_obj
            
            # Try to get a feasible solution from this assignment
            # If infeasible, repair by moving customers from overloaded facilities
            if _is_feasible(best_f, load):
                cost = _total_cost(best_f)
                if cost < best_feasible_cost:
                    best_feasible_cost = cost
                    best_feasible = best_f.copy()
                # Update upper bound
                if cost < UB:
                    UB = cost
            else:
                # Repair: move customers from overloaded facilities
                result = best_f.copy()
                load_rep = load.copy()
                # Sort customers by demand descending
                order = np.argsort(-demands)
                for c in order:
                    f = result[c]
                    if load_rep[f] > capacities[f] + 1e-6:
                        # Find best feasible facility
                        best_f_new = -1
                        best_cost = float('inf')
                        for f2 in range(n_fac):
                            if load_rep[f2] + demands[c] <= capacities[f2]:
                                c2 = assign_cost[c, f2]
                                if c2 < best_cost:
                                    best_cost = c2
                                    best_f_new = f2
                        if best_f_new >= 0:
                            load_rep[f] -= demands[c]
                            load_rep[best_f_new] += demands[c]
                            result[c] = best_f_new
                
                if _is_feasible(result, load_rep):
                    cost = _total_cost(result)
                    if cost < best_feasible_cost:
                        best_feasible_cost = cost
                        best_feasible = result.copy()
                    if cost < UB:
                        UB = cost
            
            # Update step size
            gap = UB - LB
            if gap < 1e-6:
                break
            step = (step_scale * gap) / (np.dot(lam, lam) + 1e-10)
            
            # Subgradient: load - capacity
            subgrad = load - capacities
            # Only positive violations matter
            subgrad = np.maximum(subgrad, 0)
            
            # Update multipliers
            lam = np.maximum(lam + step * subgrad, 0)
            
            # Decay step scale
            step_scale = max(0.5, step_scale * 0.98)
        
        return best_feasible, best_feasible_cost, lam

    def _local_search(result, load, deadline):
        """Best-improvement local search with Relocate and Swap moves."""
        result = result.copy()
        load = load.copy()
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            count[result[c]] += 1

        current_cost = _total_cost(result)

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # --- Phase 1: Best Relocate ---
            best_delta = 0.0
            best_c = -1
            best_f = -1

            for c in range(n_cust):
                if time.time() > deadline - 10.0:
                    break
                old_f = result[c]
                d_c = demands[c]
                base_cost = assign_cost[c, old_f]

                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c > capacities[f]:
                        continue

                    delta = assign_cost[c, f] - base_cost

                    # Open cost changes
                    if count[old_f] == 1:
                        delta -= open_cost[old_f]
                    if count[f] == 0:
                        delta += open_cost[f]

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_f = f

            if best_delta < -1e-9 and best_c != -1:
                c = best_c
                f = best_f
                old_f = result[c]
                result[c] = f
                load[old_f] -= demands[c]
                load[f] += demands[c]
                count[old_f] -= 1
                count[f] += 1
                current_cost += best_delta
                improved = True
                continue

            # --- Phase 2: Best Swap ---
            best_swap_delta = 0.0
            best_c1 = -1
            best_c2 = -1

            for c1 in range(n_cust):
                if time.time() > deadline - 10.0:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                ac1_f1 = assign_cost[c1, f1]

                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]

                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1

                    if new_load_f1 > capacities[f1] + 1e-6 or new_load_f2 > capacities[f2] + 1e-6:
                        continue

                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (ac1_f1 + assign_cost[c2, f2])

                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_c1 = c1
                        best_c2 = c2

            if best_swap_delta < -1e-9 and best_c1 != -1:
                c1, c2 = best_c1, best_c2
                f1, f2 = result[c1], result[c2]
                result[c1], result[c2] = f2, f1
                load[f1] -= demands[c1]
                load[f1] += demands[c2]
                load[f2] -= demands[c2]
                load[f2] += demands[c1]
                current_cost += best_swap_delta
                improved = True
                continue

        return result, current_cost

    # Step 1: Get initial feasible solution via greedy
    eff_cost_init = assign_cost + np.outer(demands, open_cost / np.maximum(capacities, 1.0))
    init_result, init_load = _greedy_feasible(eff_cost_init)
    init_cost = _total_cost(init_result)

    best_result = init_result.copy()
    best_cost = init_cost

    # Step 2: Lagrangian relaxation + subgradient optimization
    lag_result, lag_cost, lam = _subgradient_optimize(max_iters=150, deadline=deadline)
    
    if lag_result is not None and lag_cost < best_cost:
        best_result = lag_result.copy()
        best_cost = lag_cost

    # Step 3: Local search on the best solution found
    if best_cost == float('inf'):
        # Should not happen, but fallback
        best_result, _ = _greedy_feasible(assign_cost)
        best_cost = _total_cost(best_result)

    # Run local search on best solution
    load_best = np.zeros(n_fac)
    for c in range(n_cust):
        load_best[best_result[c]] += demands[c]
    
    refined_result, refined_cost = _local_search(best_result, load_best, deadline)
    
    if refined_cost < best_cost:
        best_result = refined_result
        best_cost = refined_cost

    return best_result

# EVOLVE-BLOCK-END
