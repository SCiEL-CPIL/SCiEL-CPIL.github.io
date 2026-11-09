# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open cost of used facilities + assignment costs."""
    used = np.unique(result)
    open_sum = float(open_cost[used].sum())
    assign_sum = float(assign_cost[np.arange(len(demands)), result].sum())
    return open_sum + assign_sum

def _is_valid(result, demands, capacities):
    """Check capacity feasibility."""
    load = np.zeros(len(capacities), dtype=float)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities + 1e-9)

def _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=0):
    """Generate an initial feasible assignment.
    mode 0: largest demand first, cheapest facility.
    mode 1: random order, cheapest facility.
    mode 2: largest demand first, cheapest facility with open-cost penalty.
    """
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    
    if mode == 0:
        order = np.argsort(-demands)
    elif mode == 1:
        order = rng.permutation(n)
    else:
        order = np.argsort(-demands)
    
    for i in order:
        # Build candidate facility list
        feasible = []
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                if mode == 2:
                    # Penalize opening a new facility
                    penalty = open_cost[f] if load[f] == 0 else 0.0
                    feasible.append((assign_cost[i, f] + penalty * 0.1, f))
                else:
                    feasible.append((assign_cost[i, f], f))
        if not feasible:
            # Should not happen if capacities are sufficient; fallback
            feasible = [(assign_cost[i, f], f) for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                # Force assign to least loaded
                f = int(np.argmin(load))
                result[i] = f
                load[f] += demands[i]
                continue
        feasible.sort()
        f = feasible[0][1]
        result[i] = f
        load[f] += demands[i]
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, rng, time_limit):
    """Improve assignment via move, swap, and consolidation moves."""
    n = len(demands)
    m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Move: reassign a single customer to a different facility
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            cur_f = result[i]
            # Try all other facilities
            for f in range(m):
                if f == cur_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Compute new cost
                old_open = open_cost[cur_f] if (load[cur_f] - demands[i] > 0) else 0.0
                new_open = open_cost[f] if load[f] == 0 else 0.0
                delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                delta_open = new_open - old_open
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost - 1e-9:
                        best_cost = current_cost
                        best_result = result.copy()
                    break
        
        # Swap: swap two customers between different facilities
        if time.time() - start_time >= time_limit:
            break
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time >= time_limit:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                # Check capacity after swap
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                # Compute cost delta
                old_open_i = open_cost[fi] if (load[fi] - demands[i] > 0) else 0.0
                new_open_i = open_cost[fi] if (load[fi] - demands[i] + demands[j] > 0) else 0.0
                old_open_j = open_cost[fj] if (load[fj] - demands[j] > 0) else 0.0
                new_open_j = open_cost[fj] if (load[fj] - demands[j] + demands[i] > 0) else 0.0
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Apply swap
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    result[i] = fj
                    result[j] = fi
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost - 1e-9:
                        best_cost = current_cost
                        best_result = result.copy()
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    # Deadline guard: keep safety margin of 10s before 150s limit
    global_start = time.time()
    deadline = global_start + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_result = None
    best_cost = np.inf
    
    # Multi-start with different seeding modes
    for restart in range(20):
        if time.time() >= deadline:
            break
        
        # Choose seeding mode
        mode = restart % 3
        
        # Generate seed
        seed_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=mode)
        
        # Validate
        if not _is_valid(seed_result, demands, capacities):
            continue
        
        seed_cost = _total_cost(seed_result, demands, capacities, open_cost, assign_cost)
        
        if seed_cost < best_cost:
            best_cost = seed_cost
            best_result = seed_result.copy()
        
        # Local search with time budget
        time_remaining = deadline - time.time()
        ls_time = min(5.0, time_remaining * 0.4)  # Up to 5s per local search
        if ls_time <= 0:
            break
        
        ls_result, ls_cost = _local_search(
            seed_result, demands, capacities, open_cost, assign_cost, rng, ls_time
        )
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_result.copy()
    
    # If no valid result found, fall back to simple greedy
    if best_result is None:
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=0)
    
    # Final validation
    if not _is_valid(best_result, demands, capacities):
        # Fallback: assign each customer to a facility with most remaining capacity
        load = np.zeros(m, dtype=float)
        best_result = np.zeros(n, dtype=int)
        for i in np.argsort(-demands):
            feasible = [f for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if feasible:
                # Pick cheapest
                f = min(feasible, key=lambda x: assign_cost[i, x])
            else:
                f = int(np.argmin(load))
            best_result[i] = f
            load[f] += demands[i]
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
