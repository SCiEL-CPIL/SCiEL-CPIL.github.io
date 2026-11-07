"""Capacitated Facility Location – Lagrangian relaxation + vectorized local search.

Strategy:
  1. Lagrangian relaxation: dualize capacity constraints, solve decomposed
     subproblems, subgradient ascent to find good multipliers.
  2. Feasibility repair from relaxed solutions.
  3. Vectorized local search (single-move + swap) with early termination.
  4. Multiple diverse restarts within time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _lagrangian_subproblem(demands, capacities, open_cost, assign_cost, lam):
    """Solve Lagrangian subproblem with open-cost-aware greedy.
    
    Returns: assignment array, dual objective value, loads per facility.
    """
    n_cust, n_fac = assign_cost.shape
    # Adjusted cost: assign_cost + lam * demand for each customer-facility pair
    adj_cost = assign_cost + lam[None, :] * demands[:, None]
    
    # Greedy assignment with open cost awareness:
    # For each customer, effective cost = adj_cost[i,f] + open_cost[f] if f not yet used
    # Process customers in order of increasing demand (smaller demands first to avoid
    # forcing large open costs)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    
    # Sort by demand ascending for greedy
    order = np.argsort(demands)
    for i in order:
        # Effective cost includes open cost for unopened facilities
        eff = adj_cost[i, :].copy()
        unopened = counts == 0
        eff[unopened] += open_cost[unopened]
        # Respect capacity
        infeasible = load + demands[i] > capacities
        eff[infeasible] = np.inf
        if np.all(np.isinf(eff)):
            # No feasible: pick facility with most remaining capacity
            rem = capacities - load
            rem[unopened] -= open_cost[unopened]  # penalize opening
            f = int(np.argmax(rem))
        else:
            f = int(np.argmin(eff))
        result[i] = f
        load[f] += demands[i]
        counts[f] += 1
    
    # Dual objective
    dual_obj = np.sum(adj_cost[np.arange(n_cust), result])
    dual_obj += np.sum(open_cost[counts > 0])
    dual_obj -= np.dot(lam, capacities)
    
    return result, dual_obj, load, counts

def _subgradient_lagrangian(demands, capacities, open_cost, assign_cost, 
                             max_iter=200, time_limit=30.0):
    """Subgradient optimization of Lagrangian dual.
    
    Returns best feasible solution found and its cost.
    """
    n_cust, n_fac = assign_cost.shape
    start_time = time.time()
    
    # Initialize multipliers
    lam = np.zeros(n_fac)
    best_dual = -np.inf
    best_result = None
    best_feasible_cost = np.inf
    
    # Step size parameters
    best_gap = np.inf
    step_scale = 1.0
    best_result_relaxed = None
    best_dual_obj = -np.inf
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        result, dual_obj, load, counts = _lagrangian_subproblem(
            demands, capacities, open_cost, assign_cost, lam)
        
        # Track best dual bound
        if dual_obj > best_dual_obj:
            best_dual_obj = dual_obj
            best_result_relaxed = result.copy()
        
        # Check feasibility of relaxed solution
        if np.all(load <= capacities + 1e-9):
            # Compute true cost
            true_cost = 0.0
            for i in range(n_cust):
                true_cost += assign_cost[i, result[i]]
            for j in range(n_fac):
                if counts[j] > 0:
                    true_cost += open_cost[j]
            if true_cost < best_feasible_cost:
                best_feasible_cost = true_cost
                best_result = result.copy()
        
        # Subgradient: load - capacities (positive where overloaded)
        grad = load - capacities
        
        # Gap for adaptive step size
        if best_result is not None:
            gap = best_feasible_cost - best_dual_obj
        else:
            gap = abs(best_dual_obj) if best_dual_obj != 0 else 1.0
        
        if gap > 1e-6:
            best_gap = min(best_gap, gap)
        
        # Polyak step size
        step = step_scale * best_gap / (np.dot(grad, grad) + 1e-12)
        step = min(step, 1.0)  # Cap step size
        
        # Update multipliers (only positive part matters for capacity)
        lam_new = lam + step * grad
        lam = np.maximum(lam_new, 0.0)  # Multipliers must be non-negative
        
        # Decay step scale periodically
        if it % 50 == 0 and it > 0:
            step_scale *= 0.7
    
    # Also try the best relaxed solution as a starting point
    if best_result_relaxed is not None:
        result = best_result_relaxed.copy()
        # Repair feasibility
        result = _repair_feasibility(demands, capacities, result)
        cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
        if cost < best_feasible_cost:
            best_feasible_cost = cost
            best_result = result.copy()
    
    if best_result is None:
        # Fallback: greedy
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
        best_feasible_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, best_result)
    
    return best_result, best_feasible_cost, best_dual_obj

def _repair_feasibility(demands, capacities, result):
    """Repair an infeasible assignment by moving customers from overloaded facilities."""
    n_cust, n_fac = len(demands), len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(20):
        overloaded = np.where(load > capacities + 1e-9)[0]
        if len(overloaded) == 0:
            break
        for j in overloaded:
            if load[j] <= capacities[j] + 1e-9:
                continue
            # Get customers at facility j, sort by demand ascending
            custs = np.where(result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                # Find facility with most remaining capacity
                rem = capacities - load
                rem[j] = -np.inf
                feasible = np.where(rem >= demands[i])[0]
                if len(feasible) > 0:
                    f = int(feasible[np.argmin(rem[feasible])])
                else:
                    f = int(np.argmax(rem))
                result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    return result

def _compute_true_cost(demands, capacities, open_cost, assign_cost, result):
    """Compute true objective value."""
    n_fac = len(capacities)
    total = 0.0
    for i in range(len(demands)):
        total += assign_cost[i, result[i]]
    counts = np.bincount(result, minlength=n_fac)
    for j in range(n_fac):
        if counts[j] > 0:
            total += open_cost[j]
    return total

def _greedy_init(demands, capacities, open_cost, assign_cost, seed=0):
    """Greedy initialization with randomization for diversity."""
    rng = np.random.RandomState(seed)
    n_cust, n_fac = len(demands), len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort by demand descending with slight randomization
    order = np.argsort(-demands + rng.uniform(0, 5, n_cust))
    
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            # Effective cost with open cost penalty
            eff = assign_cost[i, feasible].copy()
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    eff[idx] += open_cost[f] * rng.uniform(0.3, 1.0)
            f = int(feasible[np.argmin(eff)])
        result[i] = f
        load[f] += demands[i]
    
    return result

def _vectorized_local_search(demands, capacities, open_cost, assign_cost, result, 
                              time_limit=20.0):
    """Vectorized local search: compute all single-move deltas at once."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        counts[result[i]] += 1
    
    current_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
    best_cost = current_cost
    best_result = result.copy()
    
    no_improve_rounds = 0
    max_no_improve = 12
    
    while time.time() - start_time < time_limit:
        if no_improve_rounds >= max_no_improve:
            break
        
        improved = False
        # Process customers in random order for diversity
        order = np.random.permutation(n_cust)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            cur_f = result[i]
            d_i = demands[i]
            
            # Compute delta for moving customer i to each facility f
            # delta[f] = assign_cost[i,f] - assign_cost[i,cur_f]
            #           + open_cost[f] if counts[f]==0
            #           - open_cost[cur_f] if counts[cur_f]==1
            delta = assign_cost[i, :] - assign_cost[i, cur_f]
            
            # Add open cost for facilities that would be newly opened
            new_open = (counts == 0)
            delta[new_open] += open_cost[new_open]
            
            # Subtract open cost if current facility becomes empty
            if counts[cur_f] == 1:
                delta -= open_cost[cur_f]
            
            # Mask: can't stay at same facility, must have capacity
            delta[cur_f] = np.inf
            infeasible = load + d_i > capacities
            delta[infeasible] = np.inf
            
            best_f = int(np.argmin(delta))
            if delta[best_f] < -1e-9:
                # Apply move
                old_f = cur_f
                result[i] = best_f
                load[old_f] -= d_i
                load[best_f] += d_i
                counts[old_f] -= 1
                counts[best_f] += 1
                current_cost += delta[best_f]
                improved = True
        
        if improved:
            no_improve_rounds = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        else:
            no_improve_rounds += 1
    
    # If best is better than current, return best
    if best_cost < current_cost:
        return best_result, best_cost
    return result, current_cost

def _swap_local_search(demands, capacities, open_cost, assign_cost, result,
                        time_limit=10.0):
    """Swap-based local search for additional improvement."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        counts[result[i]] += 1
    
    current_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, result)
    
    # Group customers by facility
    fac_to_custs = {j: [] for j in range(n_fac)}
    for i in range(n_cust):
        fac_to_custs[result[i]].append(i)
    
    # Precompute cost matrix differences for speed
    improved_any = True
    while improved_any and time.time() - start_time < time_limit:
        improved_any = False
        # Try swaps between pairs of facilities
        for fi in range(n_fac):
            if time.time() - start_time > time_limit:
                break
            if len(fac_to_custs[fi]) == 0:
                continue
            for fj in range(fi + 1, n_fac):
                if time.time() - start_time > time_limit:
                    break
                if len(fac_to_custs[fj]) == 0:
                    continue
                
                # Vectorized: compute delta for all pairs at once
                custs_fi = np.array(fac_to_custs[fi])
                custs_fj = np.array(fac_to_custs[fj])
                
                # For each pair (i from fi, j from fj):
                # delta = assign_cost[i,fj] - assign_cost[i,fi] + assign_cost[j,fi] - assign_cost[j,fj]
                # Vectorized: shape (len_fi, len_fj)
                delta_matrix = (assign_cost[custs_fi, fj] - assign_cost[custs_fi, fi])[:, None] + \
                               (assign_cost[custs_fj, fi] - assign_cost[custs_fj, fj])[None, :]
                
                # Capacity check
                # After swap: load[fi] - demands[i] + demands[j] <= capacities[fi]
                #             load[fj] - demands[j] + demands[i] <= capacities[fj]
                cap_fi_ok = (demands[custs_fj][None, :] - demands[custs_fi][:, None]) <= \
                            (capacities[fi] - load[fi])
                cap_fj_ok = (demands[custs_fi][:, None] - demands[custs_fj][None, :]) <= \
                            (capacities[fj] - load[fj])
                
                # Open cost delta: since both facilities have customers before and after swap
                # (counts don't change: fi loses i gains j, fj loses j gains i)
                # So open cost delta is 0 unless a facility had exactly 1 customer
                oc_delta = 0.0
                # If counts[fi] == 1, after swap it still has 1 customer (lost i, gained j)
                # Same for fj. So no open cost change.
                
                feasible = cap_fi_ok & cap_fj_ok
                if np.any(feasible):
                    best_val = np.where(feasible, delta_matrix, np.inf).min()
                    if best_val < -1e-9:
                        # Find the best pair
                        idx = np.argwhere(feasible & (delta_matrix == best_val))[0]
                        i = custs_fi[idx[0]]
                        j = custs_fj[idx[1]]
                        
                        # Apply swap
                        result[i] = fj
                        result[j] = fi
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        # counts unchanged
                        current_cost += best_val
                        fac_to_custs[fi].remove(i)
                        fac_to_custs[fi].append(j)
                        fac_to_custs[fj].remove(j)
                        fac_to_custs[fj].append(i)
                        improved_any = True
                        break
                if improved_any:
                    break
            if improved_any:
                break
    
    return result, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust, n_fac = len(demands), len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Lagrangian relaxation (25s)
    lag_time = min(25.0, deadline - time.time() - 5.0)
    if lag_time > 2.0:
        result, cost, dual = _subgradient_lagrangian(
            demands, capacities, open_cost, assign_cost,
            max_iter=150, time_limit=lag_time)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 2: Multiple restarts with vectorized local search
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0 and best_result is None:
        # Fallback greedy
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=42)
        best_cost = _compute_true_cost(demands, capacities, open_cost, assign_cost, best_result)
    
    # Multiple diverse restarts
    n_restarts = max(3, int(remaining / 12.0))
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Create diverse starting point
        if r == 0 and best_result is not None:
            start = best_result.copy()
        else:
            start = _greedy_init(demands, capacities, open_cost, assign_cost, seed=r * 137 + 7)
        
        # Vectorized local search
        time_per_ls = max(4.0, (deadline - time.time() - 5.0) / max(1, n_restarts - r))
        result, cost = _vectorized_local_search(
            demands, capacities, open_cost, assign_cost, start,
            time_limit=min(time_per_ls, 30.0))
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
        
        # Additional swap-based refinement on best
        if time.time() < deadline - 8.0:
            swap_time = min(6.0, deadline - time.time() - 5.0)
            result, cost = _swap_local_search(
                demands, capacities, open_cost, assign_cost, best_result,
                time_limit=swap_time)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    # Final feasibility check and repair
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=0)
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    for j in range(n_fac):
        if load[j] > capacities[j] + 1e-9:
            custs = np.where(best_result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                rem = capacities - load
                rem[j] = -np.inf
                f = int(np.argmax(rem))
                if rem[f] >= demands[i]:
                    best_result[i] = f
                    load[j] -= demands[i]
                    load[f] += demands[i]
    
    best_result = best_result.astype(int)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
    
    return best_result
# EVOLVE-BLOCK-END
