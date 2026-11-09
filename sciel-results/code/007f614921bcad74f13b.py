# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost: open cost + assignment cost."""
    used = np.zeros(len(capacities), dtype=bool)
    for f in result:
        used[f] = True
    open_total = open_cost[used].sum()
    assign_total = assign_cost[np.arange(len(demands)), result].sum()
    return open_total + assign_total


def _greedy_assign(demands, capacities, assign_cost, allowed, order=None):
    """
    Greedy assignment: assign each customer to cheapest feasible facility.
    Returns (result, cost) or (None, inf) if infeasible.
    """
    n_cust = len(demands)
    allowed = np.array(allowed)
    n_fac = len(allowed)
    
    if sum(capacities[allowed]) < sum(demands):
        return None, float('inf')
    
    if order is None:
        order = np.argsort(-demands)
    
    costs_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_allowed, axis=1)
    
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1)
    
    for i in order:
        placed = False
        for f_local in sorted_facs[i]:
            if load[f_local] + demands[i] <= capacities[allowed[f_local]]:
                load[f_local] += demands[i]
                result[i] = allowed[f_local]
                placed = True
                break
        if not placed:
            return None, float('inf')
    
    cost = assign_cost[np.arange(n_cust), result].sum()
    return result, cost


def _local_search(result, demands, capacities, assign_cost, open_cost, deadline, max_iters=20000):
    """
    Local search with single moves and 2-swaps. Also tries facility swaps
    to potentially close expensive facilities.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Build load
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Track which facilities are open
    open_mask = load > 0
    
    best_total = _total_cost(result, demands, capacities, open_cost, assign_cost)
    
    # Precompute: for each customer, sorted facility indices by cost
    sorted_facs_all = np.argsort(assign_cost, axis=1)
    
    improved = True
    iters = 0
    stall_count = 0
    
    while improved and iters < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Single moves
        for i in range(n_cust):
            ci = result[i]
            # Try moving to a cheaper facility with capacity
            for f in sorted_facs_all[i]:
                if f == ci:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    delta = assign_cost[i, f] - assign_cost[i, ci]
                    # Check if this closes the current facility (saves open cost)
                    new_open_saving = 0
                    if load[ci] == demands[i]:  # would close facility ci
                        new_open_saving = open_cost[ci]
                    # Check if this opens a new facility
                    new_open_cost = 0
                    if load[f] == 0:
                        new_open_cost = open_cost[f]
                    
                    total_delta = delta - new_open_saving + new_open_cost
                    if total_delta < -1e-9:
                        load[ci] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        best_total += total_delta
                        improved = True
                        break
        
        # 2-swaps (sampled)
        if not improved:
            n_swaps = min(200, n_cust * (n_cust - 1) // 2)
            for _ in range(n_swaps):
                i = np.random.randint(0, n_cust)
                j = np.random.randint(0, n_cust)
                if i == j:
                    continue
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                
                # Check feasibility
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                    continue
                
                # Compute cost delta
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                               (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                delta_open = 0
                if load[fi] == demands[i] and new_load_fj > 0:  # fi closes
                    delta_open -= open_cost[fi]
                if load[fj] == demands[j] and new_load_fi > 0:  # fj closes
                    delta_open -= open_cost[fj]
                if load[fi] == demands[i] and load[fj] == demands[j]:
                    # Both close, but new assignments go to each other's facilities
                    # Actually they swap, so both still have customers
                    pass
                # More careful: after swap, fi has load new_load_fi, fj has new_load_fj
                # If new_load_fi == 0, fi closes (but we checked new_load_fi > 0 implicitly via feasibility)
                # Actually new_load_fi could be 0 if demands[j] == 0, but demands >= 5
                # So new_load_fi >= demands[j] >= 5 > 0, same for new_load_fj
                # So no facilities close in a swap where both had only one customer being swapped
                # Wait, that's not right. If fi had only customer i, and fj had only customer j,
                # after swap fi has customer j, fj has customer i. Both still open.
                # If fi had only customer i, and fj had other customers, after swap fi has j, still open.
                # So in a 2-swap, no facility closes unless one of the customers had demand 0.
                # Since demands >= 5, no facility closes in a 2-swap.
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    best_total += total_delta
                    improved = True
                    break
        
        # Facility-level moves: try moving all customers from one facility to others
        # This can close expensive facilities
        if not improved and iters % 5 == 0:
            open_facs = np.where(load > 0)[0]
            if len(open_facs) > 1:
                # Try closing the most expensive open facility
                open_facs_sorted = open_facs[np.argsort(-open_cost[open_facs])]
                for f_close in open_facs_sorted[:3]:  # try top 3
                    if time.time() > deadline:
                        break
                    customers_here = np.where(result == f_close)[0]
                    if len(customers_here) == 0:
                        continue
                    
                    # Try to move all these customers to other facilities
                    save_load = load.copy()
                    save_result = result.copy()
                    success = True
                    new_cost_delta = -open_cost[f_close]  # save open cost
                    
                    # Sort by demand descending
                    cust_order = customers_here[np.argsort(-demands[customers_here])]
                    
                    for c in cust_order:
                        placed = False
                        for f in sorted_facs_all[c]:
                            if f == f_close:
                                continue
                            if load[f] + demands[c] <= capacities[f]:
                                # Check if opening this facility costs
                                if load[f] == 0:
                                    new_cost_delta += open_cost[f]
                                new_cost_delta += assign_cost[c, f] - assign_cost[c, f_close]
                                load[f] += demands[c]
                                result[c] = f
                                placed = True
                                break
                        if not placed:
                            success = False
                            break
                    
                    if success:
                        load[f_close] = 0
                        if new_cost_delta < -1e-9:
                            best_total += new_cost_delta
                            improved = True
                            break
                    else:
                        load = save_load
                        result = save_result
    
    return result, best_total


def _simulated_annealing(result, demands, capacities, assign_cost, open_cost, deadline, n_iterations=50000):
    """
    Simulated annealing to escape local optima.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_result = result.copy()
    best_total = _total_cost(result, demands, capacities, open_cost, assign_cost)
    current_total = best_total
    
    T_start = 50.0
    T_end = 0.1
    alpha = (T_end / T_start) ** (1.0 / max(1, n_iterations))
    
    T = T_start
    sorted_facs_all = np.argsort(assign_cost, axis=1)
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        T *= alpha
        
        # Generate a random move
        move_type = np.random.random()
        
        if move_type < 0.7:
            # Single move
            i = np.random.randint(0, n_cust)
            ci = result[i]
            # Pick a random feasible facility
            feasible = np.where((load + demands[i]) <= capacities)[0]
            feasible = feasible[feasible != ci]
            if len(feasible) == 0:
                continue
            f_new = feasible[np.random.randint(0, len(feasible))]
            
            # Compute delta
            delta = assign_cost[i, f_new] - assign_cost[i, ci]
            if load[ci] == demands[i]:
                delta -= open_cost[ci]
            if load[f_new] == 0:
                delta += open_cost[f_new]
            
            # Accept
            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
                load[ci] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_total += delta
                
                if current_total < best_total:
                    best_total = current_total
                    best_result = result.copy()
                    
        elif move_type < 0.9:
            # 2-swap
            i = np.random.randint(0, n_cust)
            j = np.random.randint(0, n_cust)
            if i == j:
                continue
            fi = result[i]
            fj = result[j]
            if fi == fj:
                continue
            
            new_load_fi = load[fi] - demands[i] + demands[j]
            new_load_fj = load[fj] - demands[j] + demands[i]
            
            if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                continue
            
            delta = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                    (assign_cost[i, fi] + assign_cost[j, fj])
            
            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
                load[fi] = new_load_fi
                load[fj] = new_load_fj
                result[i] = fj
                result[j] = fi
                current_total += delta
                
                if current_total < best_total:
                    best_total = current_total
                    best_result = result.copy()
        
        else:
            # Facility swap: move all customers from one facility to another
            open_facs = np.where(load > 0)[0]
            if len(open_facs) < 2:
                continue
            f1 = open_facs[np.random.randint(0, len(open_facs))]
            f2 = open_facs[np.random.randint(0, len(open_facs))]
            if f1 == f2:
                continue
            
            customers_here = np.where(result == f1)[0]
            if len(customers_here) == 0:
                continue
            
            save_load = load.copy()
            save_result = result.copy()
            success = True
            delta = 0
            
            cust_order = customers_here[np.argsort(-demands[customers_here])]
            
            for c in cust_order:
                placed = False
                # Try f2 first, then random others
                for f in [f2] + list(np.random.choice(open_facs, size=min(5, len(open_facs)), replace=False)):
                    if f == f1:
                        continue
                    if load[f] + demands[c] <= capacities[f]:
                        if load[f] == 0:
                            delta += open_cost[f]
                        delta += assign_cost[c, f] - assign_cost[c, f1]
                        load[f] += demands[c]
                        result[c] = f
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            
            if success:
                load[f1] = 0
                delta -= open_cost[f1]
                
                if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_total += delta
                    if current_total < best_total:
                        best_total = current_total
                        best_result = result.copy()
                else:
                    load = save_load
                    result = save_result
            else:
                load = save_load
                result = save_result
    
    return best_result, best_total


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses multi-start greedy + local search + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute
    avg_assign = assign_cost.mean(axis=0)
    
    # Generate diverse starting solutions
    best_result = None
    best_total = float('inf')
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        np.argsort(-demands),  # largest first
        np.argsort(demands),   # smallest first
        np.argsort(assign_cost.mean(axis=1)),  # cheapest overall first
        np.random.permutation(n_cust),
        np.random.permutation(n_cust),
    ]
    
    for order in orderings:
        if time.time() > deadline - 60:
            break
        result, cost = _greedy_assign(demands, capacities, assign_cost, range(n_fac), order)
        if result is not None:
            total = _total_cost(result, demands, capacities, open_cost, assign_cost)
            if total < best_total:
                best_total = total
                best_result = result.copy()
    
    # Strategy 2: Greedy with facility subsets (to encourage fewer open facilities)
    by_open = np.argsort(open_cost)
    by_assign = np.argsort(avg_assign)
    by_combined = np.argsort(open_cost + avg_assign)
    
    subsets = []
    for k in [6, 8, 10, 12, 15]:
        if k <= n_fac:
            subsets.append(by_open[:k].tolist())
            subsets.append(by_assign[:k].tolist())
            subsets.append(by_combined[:k].tolist())
    
    # Remove duplicates
    seen = set()
    unique_subsets = []
    for s in subsets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    for subset in unique_subsets:
        if time.time() > deadline - 30:
            break
        result, cost = _greedy_assign(demands, capacities, assign_cost, subset, np.argsort(-demands))
        if result is not None:
            total = _total_cost(result, demands, capacities, open_cost, assign_cost)
            if total < best_total:
                best_total = total
                best_result = result.copy()
    
    # Local search on best
    if best_result is not None and time.time() < deadline - 15:
        best_result, best_total = _local_search(
            best_result, demands, capacities, assign_cost, open_cost,
            deadline=deadline - 10, max_iters=15000
        )
    
    # Simulated annealing to escape local optima
    if best_result is not None and time.time() < deadline - 5:
        best_result, best_total = _simulated_annealing(
            best_result, demands, capacities, assign_cost, open_cost,
            deadline=deadline - 2, n_iterations=30000
        )
    
    # Final local search polish
    if best_result is not None and time.time() < deadline - 1:
        best_result, best_total = _local_search(
            best_result, demands, capacities, assign_cost, open_cost,
            deadline=deadline - 0.5, max_iters=5000
        )
    
    # Fallback
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
