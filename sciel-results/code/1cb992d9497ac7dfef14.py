"""Capacitated facility location: multi-start greedy + local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_solve(demands, capacities, open_cost, assign_cost, order):
    """Greedy assignment given a customer ordering. Returns (result, cost) or (None, inf)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    opened = np.zeros(n_fac, dtype=bool)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities, sort by assign_cost
        feasible = []
        for f in range(n_fac):
            if load[f] + d <= capacities[f]:
                feasible.append(f)
        if not feasible:
            return None, np.inf
        # Sort by assign cost
        feasible_sorted = sorted(feasible, key=lambda f: assign_cost[i, f])
        # Pick the one with lowest total cost (assign + potential opening)
        best_f = None
        best_cost = np.inf
        for f in feasible_sorted:
            cost = assign_cost[i, f]
            if not opened[f]:
                cost += open_cost[f]
            if cost < best_cost:
                best_cost = cost
                best_f = f
        if best_f is None:
            return None, np.inf
        result[i] = best_f
        load[best_f] += d
        opened[best_f] = True
    
    # Compute actual cost
    total_cost = 0.0
    for f in range(n_fac):
        if opened[f]:
            total_cost += open_cost[f]
    for i in range(n_cust):
        total_cost += assign_cost[i, result[i]]
    return result, total_cost


def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
    """Improve solution via local search. Returns (best_result, best_cost)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Track loads and opened status
    load = np.zeros(n_fac, dtype=float)
    opened = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > 0:
            opened[f] = True
    
    # Compute current cost
    def compute_cost(res):
        total = 0.0
        for f in range(n_fac):
            if any(res == f):
                total += open_cost[f]
        total += np.sum(assign_cost[np.arange(n_cust), res])
        return total
    
    best_cost = compute_cost(result)
    best_result = result.copy()
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # --- 1-move: reassign a single customer ---
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            cur_f = result[i]
            d = demands[i]
            for f in range(n_fac):
                if f == cur_f:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                # Cost change: 
                # - remove assign_cost[i, cur_f]
                # + add assign_cost[i, f]
                # if cur_f becomes empty, subtract open_cost[cur_f]
                # if f was not opened, add open_cost[f]
                delta = assign_cost[i, f] - assign_cost[i, cur_f]
                new_load_cur = load[cur_f] - d
                new_load_f = load[f] + d
                if new_load_cur <= 0 and opened[cur_f]:
                    delta -= open_cost[cur_f]
                if new_load_f <= 0 and not opened[f]:
                    pass  # f was not opened, now it will be
                if new_load_f > 0 and not opened[f]:
                    delta += open_cost[f]
                
                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= d
                    load[f] += d
                    result[i] = f
                    if new_load_cur <= 0:
                        opened[cur_f] = False
                    if new_load_f > 0:
                        opened[f] = True
                    best_cost += delta
                    best_result = result.copy()
                    improved = True
                    break  # Re-evaluate from scratch after a move
        
        # --- 2-move: swap two customers between facilities ---
        if time.time() - start_time > time_limit:
            break
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_cust):
                if time.time() - start_time > time_limit:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                di = demands[i]
                dj = demands[j]
                # After swap: i goes to fj, j goes to fi
                if load[fi] - di + dj > capacities[fi]:
                    continue
                if load[fj] - dj + di > capacities[fj]:
                    continue
                
                # Cost change
                delta = (assign_cost[i, fj] - assign_cost[i, fi]) + (assign_cost[j, fi] - assign_cost[j, fj])
                
                # Opening cost changes
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                if new_load_fi <= 0 and opened[fi]:
                    delta -= open_cost[fi]
                if new_load_fj <= 0 and opened[fj]:
                    delta -= open_cost[fj]
                if new_load_fi > 0 and not opened[fi]:
                    delta += open_cost[fi]
                if new_load_fj > 0 and not opened[fj]:
                    delta += open_cost[fj]
                
                if delta < -1e-9:
                    # Apply swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    if new_load_fi <= 0:
                        opened[fi] = False
                    if new_load_fj <= 0:
                        opened[fj] = False
                    if new_load_fi > 0:
                        opened[fi] = True
                    if new_load_fj > 0:
                        opened[fj] = True
                    best_cost += delta
                    best_result = result.copy()
                    improved = True
                    break
            if improved:
                break
    
    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    overall_deadline = 110.0  # seconds, safety margin before 120s
    start_time = time.time()
    
    # Phase 1: Multi-start greedy to get a good initial solution
    best_result = None
    best_cost = np.inf
    
    # Try several orderings
    orderings = []
    # Descending demand
    orderings.append(np.argsort(-demands))
    # Ascending demand
    orderings.append(np.argsort(demands))
    # By average assignment cost
    avg_cost = assign_cost.mean(axis=1)
    orderings.append(np.argsort(avg_cost))
    # A few random orderings
    rng = np.random.RandomState(42)
    for _ in range(5):
        orderings.append(rng.permutation(n_cust))
    
    for order in orderings:
        if time.time() - start_time > overall_deadline - 30:
            break
        res, cost = _greedy_solve(demands, capacities, open_cost, assign_cost, order)
        if res is not None and cost < best_cost:
            best_cost = cost
            best_result = res.copy()
    
    if best_result is None:
        # Fallback: simple greedy
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result
    
    # Phase 2: Local search
    remaining_time = overall_deadline - (time.time() - start_time)
    if remaining_time > 2.0:
        best_result, best_cost = _local_search(
            best_result, demands, capacities, open_cost, assign_cost, 
            remaining_time - 2.0
        )
    
    # Validate
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-6:
            # Should not happen, but fallback
            return np.zeros(n_cust, dtype=int)
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
