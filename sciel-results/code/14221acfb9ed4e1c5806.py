"""Capacitated Facility Location – LP relaxation + vectorized local search + facility subset search.

Strategy:
  1. LP relaxation via HiGHS to get a high-quality fractional solution.
  2. Rounding + vectorized local search to get feasible integer solution.
  3. Facility subset search to explore different open sets.
  4. Pair-exchange polish.
  5. Multiple diverse restarts within time budget.
"""
import numpy as np
import time
from scipy.optimize import linprog

def _lp_relaxation(demands, capacities, open_cost, assign_cost):
    """Solve LP relaxation of CFLP using HiGHS.
    
    Variables:
      x[i,j] >= 0: fraction of customer i assigned to facility j
      y[j] >= 0: facility j open (binary in IP, continuous in LP)
    
    Constraints:
      sum_j x[i,j] = 1 for all i
      sum_i demands[i] * x[i,j] <= capacities[j] * y[j] for all j
      0 <= x[i,j] <= 1
      0 <= y[j] <= 1
    
    Objective:
      min sum_{i,j} assign_cost[i,j] * x[i,j] + sum_j open_cost[j] * y[j]
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variable ordering: x[0,0], x[0,1], ..., x[0,n_fac-1], x[1,0], ..., x[n_cust-1,n_fac-1], y[0], ..., y[n_fac-1]
    n_vars = n_cust * n_fac + n_fac
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c[n_cust * n_fac + j] = open_cost[j]
    
    # Equality constraints: sum_j x[i,j] = 1 for each customer
    A_eq = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    b_eq = np.ones(n_cust)
    
    # Inequality constraints: sum_i demands[i] * x[i,j] <= capacities[j] * y[j]
    # Rearranged: sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    A_ub = np.zeros((n_fac, n_vars))
    for j in range(n_fac):
        for i in range(n_cust):
            A_ub[j, i * n_fac + j] = demands[i]
        A_ub[j, n_cust * n_fac + j] = -capacities[j]
    b_ub = np.zeros(n_fac)
    
    # Bounds
    bounds = [(0, 1)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method='highs')
        if not res.success:
            return None, None
        
        x = np.zeros((n_cust, n_fac))
        for i in range(n_cust):
            for j in range(n_fac):
                x[i, j] = res.x[i * n_fac + j]
        y = res.x[n_cust * n_fac:]
        
        return x, y
    except Exception:
        return None, None


def _round_lp_solution(x, y, demands, capacities, open_cost, assign_cost):
    """Round LP solution to a feasible integer assignment."""
    n_cust, n_fac = x.shape
    
    # Sort customers by LP "certainty" (max x[i,j]) descending
    max_x = x.max(axis=1)
    order = np.argsort(-max_x)
    
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in order:
        # Find best facility based on LP solution, then cost
        scores = assign_cost[i, :] - 50.0 * x[i, :]  # LP guidance
        feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            f = int(feasible[np.argmin(scores[feasible])])
        result[i] = f
        load[f] += demands[i]
    
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=20.0):
    """Vectorized single-move local search with batch processing."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = float(np.sum(assign_cost[np.arange(n_cust), result]))
    for j in range(n_fac):
        if counts[j] > 0:
            current_cost += open_cost[j]
    
    best_cost = current_cost
    best_result = result.copy()
    no_improve = 0
    max_no_improve = 6
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        order = np.random.permutation(n_cust)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            cur_f = result[i]
            d_i = demands[i]
            
            # Compute delta for moving i to each facility (vectorized)
            delta = assign_cost[i, :] - assign_cost[i, cur_f]
            
            # Open cost adjustments
            if counts[cur_f] == 1:
                delta -= open_cost[cur_f]
            empty_facs = np.where(counts == 0)[0]
            if len(empty_facs) > 0:
                delta[empty_facs] += open_cost[empty_facs]
            
            # Capacity constraint
            delta[cur_f] = np.inf
            infeasible = (load + d_i) > (capacities + 1e-9)
            delta[infeasible] = np.inf
            
            best_f = int(np.argmin(delta))
            if delta[best_f] < -1e-9:
                old_f = cur_f
                result[i] = best_f
                load[old_f] -= d_i
                load[best_f] += d_i
                counts[old_f] -= 1
                counts[best_f] += 1
                current_cost += delta[best_f]
                improved = True
        
        if improved:
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        else:
            no_improve += 1
    
    if best_cost < current_cost - 1e-9:
        return best_result, best_cost
    return result, current_cost


def _pair_exchange(demands, capacities, open_cost, assign_cost, result, time_limit=8.0):
    """Pair-exchange local search with vectorized cost computation."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = float(np.sum(assign_cost[np.arange(n_cust), result]))
    for j in range(n_fac):
        if counts[j] > 0:
            current_cost += open_cost[j]
    
    improved_any = True
    max_rounds = 20
    rnd = 0
    
    while improved_any and time.time() - start_time < time_limit and rnd < max_rounds:
        rnd += 1
        improved_any = False
        
        n_pairs = min(2000, n_cust * (n_cust - 1) // 2)
        for _ in range(n_pairs):
            if time.time() - start_time > time_limit:
                break
            
            i = np.random.randint(n_cust)
            j = np.random.randint(n_cust)
            if i == j:
                continue
            
            fi = result[i]
            fj = result[j]
            if fi == fj:
                continue
            
            if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                continue
            if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                continue
            
            delta = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                    (assign_cost[j, fi] - assign_cost[j, fj])
            
            if delta < -1e-9:
                result[i] = fj
                result[j] = fi
                load[fi] += demands[j] - demands[i]
                load[fj] += demands[i] - demands[j]
                current_cost += delta
                improved_any = True
    
    return result, current_cost


def _greedy_init(demands, capacities, open_cost, assign_cost, seed=0):
    """Greedy initialization with open-cost awareness."""
    rng = np.random.RandomState(seed)
    n_cust, n_fac = len(demands), len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    order = np.argsort(-demands + rng.uniform(0, 3, n_cust))
    
    for i in order:
        feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            eff = assign_cost[i, feasible].copy()
            empty_mask = load[feasible] == 0
            if np.any(empty_mask):
                eff[empty_mask] += open_cost[feasible[empty_mask]] * rng.uniform(0.3, 0.8)
            f = int(feasible[np.argmin(eff)])
        result[i] = f
        load[f] += demands[i]
    
    return result


def _facility_subset_search(demands, capacities, open_cost, assign_cost, 
                            current_best, time_limit=20.0):
    """Try different facility subsets to find good solutions."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    
    best_result = current_best.copy()
    best_cost = float(np.sum(assign_cost[np.arange(n_cust), best_result]))
    counts = np.bincount(best_result, minlength=n_fac)
    for j in range(n_fac):
        if counts[j] > 0:
            best_cost += open_cost[j]
    
    open_set = set(np.where(counts > 0)[0].tolist())
    
    # Try removing each open facility (sorted by open cost descending)
    for j in sorted(open_set, key=lambda x: -open_cost[x]):
        if time.time() - start_time > time_limit:
            break
        
        trial_set = open_set - {j}
        if len(trial_set) == 0:
            continue
        
        total_cap = sum(capacities[f] for f in trial_set)
        if total_cap < sum(demands) - 1e-9:
            continue
        
        trial = best_result.copy()
        load = np.zeros(n_fac)
        np.add.at(load, trial, demands)
        
        custs = np.where(trial == j)[0]
        custs = custs[np.argsort(demands[custs])]
        feasible = True
        for i in custs:
            rem = [f for f in trial_set if load[f] + demands[i] <= capacities[f] + 1e-9]
            if not rem:
                feasible = False
                break
            costs = assign_cost[i, rem]
            best_f = rem[np.argmin(costs)]
            trial[i] = best_f
            load[best_f] += demands[i]
        
        if not feasible:
            continue
        
        new_cost = float(np.sum(assign_cost[np.arange(n_cust), trial]))
        new_counts = np.bincount(trial, minlength=n_fac)
        for f in trial_set:
            if new_counts[f] > 0:
                new_cost += open_cost[f]
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_result = trial.copy()
            open_set = trial_set
    
    # Try adding each closed facility
    closed_set = set(range(n_fac)) - open_set
    for j in sorted(closed_set, key=lambda x: open_cost[x]):
        if time.time() - start_time > time_limit:
            break
        
        trial = best_result.copy()
        load = np.zeros(n_fac)
        np.add.at(load, trial, demands)
        
        for i in range(n_cust):
            if trial[i] == j:
                continue
            if load[j] + demands[i] > capacities[j] + 1e-9:
                continue
            delta = assign_cost[i, j] - assign_cost[i, trial[i]]
            if delta < -1e-9:
                old_f = trial[i]
                trial[i] = j
                load[old_f] -= demands[i]
                load[j] += demands[i]
        
        new_cost = float(np.sum(assign_cost[np.arange(n_cust), trial]))
        new_counts = np.bincount(trial, minlength=n_fac)
        for f in new_counts:
            if new_counts[f] > 0:
                new_cost += open_cost[f]
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_result = trial.copy()
            open_set.add(j)
    
    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust, n_fac = len(demands), len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: LP relaxation for high-quality initialization
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        lp_time = min(15.0, remaining * 0.4)
        x_lp, y_lp = _lp_relaxation(demands, capacities, open_cost, assign_cost)
        if x_lp is not None:
            result = _round_lp_solution(x_lp, y_lp, demands, capacities, open_cost, assign_cost)
            time_left = deadline - time.time() - 5.0
            if time_left > 5.0:
                result, cost = _local_search(demands, capacities, open_cost, assign_cost, result,
                                             time_limit=min(20.0, time_left * 0.5))
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.copy()
    
    # Phase 2: Greedy initialization + local search (diverse start)
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=42)
        time_left = deadline - time.time() - 5.0
        if time_left > 5.0:
            result, cost = _local_search(demands, capacities, open_cost, assign_cost, result,
                                         time_limit=min(20.0, time_left * 0.5))
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    # Phase 3: Facility subset search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0 and best_result is not None:
        result, cost = _facility_subset_search(demands, capacities, open_cost, assign_cost,
                                               best_result, time_limit=min(15.0, remaining * 0.4))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 4: Pair exchange polish
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0 and best_result is not None:
        result, cost = _pair_exchange(demands, capacities, open_cost, assign_cost, best_result,
                                      time_limit=min(10.0, remaining * 0.3))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 5: More local search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0 and best_result is not None:
        result, cost = _local_search(demands, capacities, open_cost, assign_cost, best_result,
                                     time_limit=min(15.0, remaining * 0.4))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 6: Diverse restarts
    remaining = deadline - time.time() - 10.0
    if remaining > 10.0:
        n_restarts = max(1, int(remaining / 20.0))
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            start = _greedy_init(demands, capacities, open_cost, assign_cost, 
                                seed=r * 137 + 7 + 1000)
            
            time_left = deadline - time.time() - 10.0
            if time_left < 5.0:
                break
            
            result, cost = _local_search(demands, capacities, open_cost, assign_cost, start,
                                         time_limit=min(10.0, time_left * 0.5))
            
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    # Final polish
    remaining = deadline - time.time() - 3.0
    if remaining > 3.0 and best_result is not None:
        result, cost = _local_search(demands, capacities, open_cost, assign_cost, best_result,
                                     time_limit=min(8.0, remaining * 0.8))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Final feasibility check and repair
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=0)
    
    load = np.zeros(n_fac)
    np.add.at(load, best_result, demands)
    
    for j in range(n_fac):
        if load[j] > capacities[j] + 1e-9:
            custs = np.where(best_result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                rem = capacities - load
                rem[j] = -np.inf
                feasible = np.where(rem >= demands[i])[0]
                if len(feasible) > 0:
                    f = int(feasible[np.argmin(rem[feasible])])
                else:
                    f = int(np.argmax(rem))
                best_result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    best_result = best_result.astype(int)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
    
    return best_result
