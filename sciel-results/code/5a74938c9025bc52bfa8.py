"""
Structural Change: LP-Relaxation-Guided Rounding with Multi-Start Local Search.

This approach formulates the problem as a mixed-integer program where assignment
variables are relaxed to continuous values. We use the fractional components of
the LP solution to guide a deterministic rounding procedure that respects capacity
constraints and minimizes the gap to the optimal integer solution.

Key structural ideas:
1. Solve a simplified LP relaxation via iterative price updates (Lagrangian dual)
2. Use the fractional assignment probabilities to guide rounding via a regret-based
   greedy procedure
3. Multiple restarts with different price initializations to escape local optima
4. Efficient local search with 1-moves and 2-swaps for final polishing
"""

import numpy as np
import time

DEADLINE = 115.0  # Safety margin before 150s kill


def _total_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Calculate the total cost of an assignment."""
    used = np.zeros(len(capacities), dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = float(open_cost[used].sum())
    cust = np.arange(len(assignment))
    assign_sum = float(assign_cost[cust, assignment].sum())
    return open_sum + assign_sum


def _valid(assignment, demands, capacities):
    """Check if assignment is valid (integer indices, capacity constraints)."""
    if len(assignment) != len(demands):
        return False
    if np.any(assignment < 0) or np.any(assignment >= len(capacities)):
        return False
    if np.any(np.asarray(assignment, dtype=float) % 1 != 0):
        return False
    loads = np.zeros(len(capacities))
    np.add.at(loads, assignment, demands)
    return bool(np.all(loads <= capacities + 1e-9))


def _solve_lagrangian_dual(demands, capacities, assign_cost, lam_init, max_iter=300, time_limit=8.0):
    """
    Solve the Lagrangian dual to get facility prices (lambda).
    Returns lambda array.
    """
    n_f = len(capacities)
    n_c = len(demands)
    lam = lam_init.copy()
    best_dual = -np.inf
    t_start = time.time()
    
    # Track best lambda found
    best_lam = lam.copy()
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
            
        # Subproblem: assign each customer to argmin_j (c_ij + lam_j)
        adjusted_costs = assign_cost + lam[None, :]
        best_facilities = np.argmin(adjusted_costs, axis=1)
        best_costs = adjusted_costs[np.arange(n_c), best_facilities]
        
        # Calculate load
        loads = np.zeros(n_f)
        np.add.at(loads, best_facilities, demands)
        
        # Dual value: sum of min costs + sum(lambda * capacity)
        dual_val = float(np.sum(best_costs)) + float(np.dot(lam, capacities))
        
        if dual_val > best_dual:
            best_dual = dual_val
            best_lam = lam.copy()
            
        # Subgradient: s_j = capacity_j - load_j
        subgrad = capacities - loads
        
        # Update lambda with adaptive step size
        norm_s = np.linalg.norm(subgrad)
        if norm_s < 1e-10:
            break
            
        # Use a decaying step size
        step = 2.0 / (1.0 + 0.05 * it)
        lam = np.maximum(lam + step * subgrad / max(norm_s, 1e-6), 0.0)
        
        # Occasionally perturb to escape local minima
        if it % 50 == 49:
            lam += np.random.uniform(-0.5, 0.5, size=n_f)
            lam = np.maximum(lam, 0.0)
        
    return best_lam


def _lp_guided_rounding(demands, capacities, assign_cost, lam, temperature=1.0):
    """
    Deterministic rounding procedure guided by Lagrangian prices (LP relaxation insight).
    
    The idea: In the LP relaxation, each customer is fractionally assigned to multiple
    facilities. The Lagrangian prices give us an implicit probability distribution.
    We use regret (gap between best and second-best) to determine assignment order,
    and capacity constraints to ensure feasibility.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    adjusted_costs = assign_cost + lam[None, :]
    
    # Calculate regret: difference between best and second best cost
    # Higher regret means the LP solution is more "confident" about that assignment
    sorted_costs = np.sort(adjusted_costs, axis=1)
    regrets = sorted_costs[:, 1] - sorted_costs[:, 0]
    
    # Add a small random perturbation to break ties and diversify
    if temperature > 0:
        regrets += np.random.normal(0, temperature * 0.1, size=n_c)
    
    # Order customers by regret (high regret first = more "integral" in LP sense)
    order = np.argsort(-regrets)
    
    assignment = np.full(n_c, -1, dtype=int)
    loads = np.zeros(n_f)
    
    # First pass: assign in order of regret
    for i in order:
        # Find feasible facilities
        feasible_mask = loads + demands[i] <= capacities + 1e-9
        
        if not np.any(feasible_mask):
            # No feasible facility - should not happen if total demand <= total capacity
            # Fallback: pick facility with most remaining space
            remaining = capacities - loads
            best_f = np.argmax(remaining)
            assignment[i] = best_f
            loads[best_f] += demands[i]
            continue
        
        # Among feasible facilities, pick the one with lowest adjusted cost
        feasible_costs = adjusted_costs[i].copy()
        feasible_costs[~feasible_mask] = np.inf
        best_f = np.argmin(feasible_costs)
        
        assignment[i] = best_f
        loads[best_f] += demands[i]
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Efficient local search with 1-moves and 2-swaps.
    Uses vectorized operations where possible to speed up.
    """
    n_c = len(assignment)
    n_f = len(capacities)
    
    if n_c == 0:
        return assignment
        
    assignment = assignment.copy()
    loads = np.zeros(n_f)
    np.add.at(loads, assignment, demands)
    
    in_use = np.zeros(n_f, dtype=bool)
    for f in assignment:
        in_use[f] = True
        
    t_start = time.time()
    
    # Precompute current costs for each customer
    current_costs = assign_cost[np.arange(n_c), assignment].copy()
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() >= deadline:
            break
            
        improved = False
        
        # --- 1-Move Search ---
        # For each customer, find the best alternative facility
        for i in range(n_c):
            if time.time() >= deadline:
                break
                
            cur_f = assignment[i]
            d = demands[i]
            cur_cost = current_costs[i]
            
            # Calculate cost delta for moving i to each f
            new_costs = assign_cost[i, :]
            
            # Open cost changes
            open_deltas = np.zeros(n_f)
            # If f is not in use, we pay open_cost[f]
            open_deltas[~in_use] = open_cost[~in_use]
            # If cur_f becomes unused, we save open_cost[cur_f]
            if loads[cur_f] == d:
                open_deltas -= open_cost[cur_f]
            
            total_deltas = new_costs - cur_cost + open_deltas
            
            # Mask infeasible facilities
            feasible = loads + d <= capacities + 1e-9
            feasible = feasible & (np.arange(n_f) != cur_f)
            
            if not np.any(feasible):
                continue
                
            total_deltas[~feasible] = np.inf
            
            best_f = np.argmin(total_deltas)
            gain = -total_deltas[best_f]
            
            if gain > 1e-9:
                # Perform move
                loads[cur_f] -= d
                loads[best_f] += d
                assignment[i] = best_f
                current_costs[i] = assign_cost[i, best_f]
                
                # Update in_use
                in_use[best_f] = True
                if loads[cur_f] == 0:
                    in_use[cur_f] = False
                    
                improved = True
        
        # --- 2-Swap Search ---
        if time.time() < deadline:
            for i in range(n_c):
                if time.time() >= deadline:
                    break
                fi = assignment[i]
                di = demands[i]
                ci = current_costs[i]
                
                for k in range(i + 1, n_c):
                    fk = assignment[k]
                    if fi == fk:
                        continue
                    dk = demands[k]
                    ck = current_costs[k]
                    
                    # Check capacity after swap
                    if loads[fi] - di + dk > capacities[fi] + 1e-9:
                        continue
                    if loads[fk] - dk + di > capacities[fk] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    new_ci = assign_cost[i, fk]
                    new_ck = assign_cost[k, fi]
                    
                    delta_assign = (new_ci + new_ck) - (ci + ck)
                    
                    if delta_assign < -1e-9:
                        # Perform swap
                        loads[fi] = loads[fi] - di + dk
                        loads[fk] = loads[fk] - dk + di
                        assignment[i] = fk
                        assignment[k] = fi
                        current_costs[i] = new_ci
                        current_costs[k] = new_ck
                        
                        improved = True
                        break  # Restart i loop after a swap
                
                if improved and time.time() >= deadline:
                    break
        
        if not improved:
            break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_c = len(demands)
    n_f = len(capacities)
    t0 = time.time()
    
    # Track best solution found
    best_assign = None
    best_cost = np.inf
    
    # Multiple restarts with different Lagrangian price initializations
    # This is the key structural change: multi-start guided by LP relaxation insights
    
    num_restarts = 5
    restart_budget = (DEADLINE - 10.0) / num_restarts  # ~20s per restart
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        restart_deadline = t0 + restart * restart_budget + restart_budget * 0.7
        
        # Different initialization for each restart
        if restart == 0:
            # Start with zero prices
            lam_init = np.zeros(n_f)
        elif restart == 1:
            # Start with prices proportional to open cost
            lam_init = open_cost * 0.5
        elif restart == 2:
            # Start with random prices
            lam_init = np.random.uniform(0, 20, size=n_f)
        elif restart == 3:
            # Start with prices based on capacity utilization heuristic
            avg_demand = np.mean(demands)
            lam_init = np.clip(avg_demand / (capacities + 1e-6) * 10, 0, 50)
        else:
            # Random perturbation of previous best lambda
            if restart > 0:
                lam_init = np.random.uniform(0, 30, size=n_f)
            else:
                lam_init = np.zeros(n_f)
        
        # Solve Lagrangian dual
        lam = _solve_lagrangian_dual(demands, capacities, assign_cost, lam_init, 
                                      max_iter=200, time_limit=restart_budget * 0.3)
        
        # LP-guided rounding with slight temperature for diversity
        temperature = 0.0 if restart == 0 else 0.5 * restart
        candidate = _lp_guided_rounding(demands, capacities, assign_cost, lam, 
                                         temperature=temperature)
        
        # Ensure validity
        if not _valid(candidate, demands, capacities):
            # Fallback to simple greedy
            candidate = np.zeros(n_c, dtype=int)
            loads = np.zeros(n_f)
            order = np.argsort(-demands)
            for i in order:
                for f in range(n_f):
                    if loads[f] + demands[i] <= capacities[f]:
                        candidate[i] = f
                        loads[f] += demands[i]
                        break
        
        candidate_cost = _total_cost(candidate, demands, capacities, open_cost, assign_cost)
        
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_assign = candidate.copy()
        
        # Local search polish
        remaining_for_ls = min(restart_deadline, t0 + DEADLINE - 5)
        if time.time() < remaining_for_ls - 2:
            polished = _local_search(candidate, demands, capacities, open_cost, 
                                      assign_cost, remaining_for_ls)
            if _valid(polished, demands, capacities):
                polished_cost = _total_cost(polished, demands, capacities, open_cost, 
                                            assign_cost)
                if polished_cost < best_cost:
                    best_cost = polished_cost
                    best_assign = polished.copy()
    
    # Final polish with remaining time
    if best_assign is not None and time.time() - t0 < DEADLINE - 3:
        final_polished = _local_search(best_assign, demands, capacities, open_cost, 
                                        assign_cost, t0 + DEADLINE - 2)
        if _valid(final_polished, demands, capacities):
            final_cost = _total_cost(final_polished, demands, capacities, open_cost, 
                                     assign_cost)
            if final_cost < best_cost:
                best_assign = final_polished
                best_cost = final_cost
    
    # Ensure we have a valid result
    if best_assign is None or not _valid(best_assign, demands, capacities):
        # Absolute fallback
        best_assign = np.zeros(n_c, dtype=int)
        loads = np.zeros(n_f)
        order = np.argsort(-demands)
        for i in order:
            for f in range(n_f):
                if loads[f] + demands[i] <= capacities[f]:
                    best_assign[i] = f
                    loads[f] += demands[i]
                    break
    
    return best_assign.astype(int)
