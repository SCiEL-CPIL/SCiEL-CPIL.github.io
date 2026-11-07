"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return np.inf
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    total += np.sum(assign_cost[np.arange(len(result)), result])
    return total


def _is_valid(result, demands, capacities):
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return False
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            return False
    return True


def _greedy_init(demands, capacities, open_cost, assign_cost, order=None, random_tie=False, rng=None):
    n_cust = len(demands)
    n_fac = len(capacities)
    if order is None:
        order = np.argsort(-demands)
    else:
        order = np.array(order)
    
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    
    for i in order:
        # Find all feasible facilities
        feasible = []
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                feasible.append(f)
        
        if len(feasible) == 0:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            best_f = np.argmax(rem)
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Sort feasible by assign_cost
            feasible_sorted = sorted(feasible, key=lambda f: assign_cost[i, f])
            if random_tie and rng is not None and len(feasible_sorted) > 1:
                # Randomly pick among top 3 or so to diversify
                top_k = min(3, len(feasible_sorted))
                best_f = feasible_sorted[rng.randint(top_k)]
            else:
                best_f = feasible_sorted[0]
            load[best_f] += demands[i]
            result[i] = best_f
    
    return result


def _local_search_move(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search with move operations only."""
    n_cust = len(demands)
    n_fac = len(capacities)
    demands_f = demands.astype(float)
    
    # Compute initial load
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[result[i]] += demands_f[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            cur_f = result[i]
            # Find best feasible facility
            best_new_f = -1
            best_delta = 0.0
            for f in range(n_fac):
                if f == cur_f:
                    continue
                if load[f] + demands_f[i] > capacities[f] + 1e-9:
                    continue
                delta = assign_cost[i, f] - assign_cost[i, cur_f]
                open_delta = 0.0
                if load[f] == 0:
                    open_delta += open_cost[f]
                if load[cur_f] - demands_f[i] <= 1e-9:
                    open_delta -= open_cost[cur_f]
                total_delta = delta + open_delta
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_new_f = f
            if best_new_f >= 0 and best_delta < -1e-9:
                # Apply move
                load[cur_f] -= demands_f[i]
                load[best_new_f] += demands_f[i]
                result[i] = best_new_f
                current_cost += best_delta
                improved = True
    
    return result, current_cost


def _local_search_swap(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search with swap operations between two customers."""
    n_cust = len(demands)
    n_fac = len(capacities)
    demands_f = demands.astype(float)
    
    # Compute initial load
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[result[i]] += demands_f[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cust):
                if time.time() > deadline:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                
                # Try swapping: i goes to fj, j goes to fi
                # Check capacity constraints
                # After swap: load[fi] - demands[i] + demands[j], load[fj] - demands[j] + demands[i]
                new_load_fi = load[fi] - demands_f[i] + demands_f[j]
                new_load_fj = load[fj] - demands_f[j] + demands_f[i]
                
                if new_load_fi > capacities[fi] + 1e-9 or new_load_fj > capacities[fj] + 1e-9:
                    continue
                
                # Compute cost delta
                # Old: assign_cost[i, fi] + assign_cost[j, fj]
                # New: assign_cost[i, fj] + assign_cost[j, fi]
                assign_delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost delta: 
                # fi becomes empty if new_load_fi == 0
                # fj becomes empty if new_load_fj == 0
                # fi was empty? No, since i was there
                # fj was empty? No, since j was there
                open_delta = 0.0
                if new_load_fi <= 1e-9:
                    open_delta -= open_cost[fi]
                if new_load_fj <= 1e-9:
                    open_delta -= open_cost[fj]
                # If fi was not empty before (it wasn't, since i was there) and now empty, subtract
                # If fj was not empty before (it wasn't, since j was there) and now empty, subtract
                # If fi becomes non-empty (it was non-empty), no change
                # If fj becomes non-empty (it was non-empty), no change
                
                total_delta = assign_delta + open_delta
                
                if total_delta < -1e-9:
                    # Apply swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    current_cost += total_delta
                    improved = True
    
    return result, current_cost


def _simulated_annealing(result, demands, capacities, open_cost, assign_cost, deadline, seed=42):
    """Simulated annealing with move operations."""
    n_cust = len(demands)
    n_fac = len(capacities)
    demands_f = demands.astype(float)
    
    rng = np.random.RandomState(seed)
    
    # Compute initial load
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[result[i]] += demands_f[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    best_result = result.copy()
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.5
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Pick a random customer
        i = rng.randint(n_cust)
        cur_f = result[i]
        
        # Try to move to a different facility
        candidates = [f for f in range(n_fac) if f != cur_f]
        if len(candidates) == 0:
            T *= cooling
            continue
        new_f = candidates[rng.randint(len(candidates))]
        
        # Check capacity
        if load[new_f] + demands_f[i] > capacities[new_f] + 1e-9:
            T *= cooling
            continue
        
        # Compute cost difference
        delta = (assign_cost[i, new_f] - assign_cost[i, cur_f])
        
        # Open cost change
        open_delta = 0.0
        if load[new_f] == 0:
            open_delta += open_cost[new_f]
        if load[cur_f] - demands_f[i] <= 1e-9:
            open_delta -= open_cost[cur_f]
        
        delta_total = delta + open_delta
        
        if delta_total < 0 or rng.random() < np.exp(-delta_total / T):
            # Accept
            result[i] = new_f
            load[cur_f] -= demands_f[i]
            load[new_f] += demands_f[i]
            current_cost += delta_total
            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_result = result.copy()
        
        T *= cooling
    
    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Multiple diverse initializations
    rng = np.random.RandomState(123)
    
    # 1. Largest demand first, deterministic
    result1 = _greedy_init(demands, capacities, open_cost, assign_cost, order=None, random_tie=False)
    result1, cost1 = _local_search_move(result1, demands, capacities, open_cost, assign_cost, DEADLINE)
    if cost1 < best_cost:
        best_cost = cost1
        best_result = result1.copy()
    
    if time.time() > DEADLINE:
        return best_result
    
    # 2. Largest demand first, random tie-breaking
    result2 = _greedy_init(demands, capacities, open_cost, assign_cost, order=None, random_tie=True, rng=rng)
    result2, cost2 = _local_search_move(result2, demands, capacities, open_cost, assign_cost, DEADLINE)
    if cost2 < best_cost:
        best_cost = cost2
        best_result = result2.copy()
    
    if time.time() > DEADLINE:
        return best_result
    
    # 3. Random order
    order3 = rng.permutation(n_cust)
    result3 = _greedy_init(demands, capacities, open_cost, assign_cost, order=order3, random_tie=True, rng=rng)
    result3, cost3 = _local_search_move(result3, demands, capacities, open_cost, assign_cost, DEADLINE)
    if cost3 < best_cost:
        best_cost = cost3
        best_result = result3.copy()
    
    if time.time() > DEADLINE:
        return best_result
    
    # Apply swap local search to best result
    result_swap = best_result.copy()
    result_swap, cost_swap = _local_search_swap(result_swap, demands, capacities, open_cost, assign_cost, DEADLINE)
    if cost_swap < best_cost:
        best_cost = cost_swap
        best_result = result_swap.copy()
    
    if time.time() > DEADLINE:
        return best_result
    
    # Apply simulated annealing to best result
    result_sa = best_result.copy()
    result_sa, cost_sa = _simulated_annealing(result_sa, demands, capacities, open_cost, assign_cost, DEADLINE, seed=42)
    if cost_sa < best_cost:
        best_cost = cost_sa
        best_result = result_sa.copy()
    
    if time.time() > DEADLINE:
        return best_result
    
    # Final move local search
    result_final = best_result.copy()
    result_final, cost_final = _local_search_move(result_final, demands, capacities, open_cost, assign_cost, DEADLINE)
    if cost_final < best_cost:
        best_cost = cost_final
        best_result = result_final.copy()
    
    # Ensure validity
    if not _is_valid(best_result, demands, capacities):
        # Fall back to greedy
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, order=None, random_tie=False)
    
    return best_result
# EVOLVE-BLOCK-END
