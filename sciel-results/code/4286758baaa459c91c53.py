"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _total_cost(demands, capacities, open_cost, assign_cost, assignment):
    load = np.zeros(len(capacities), dtype=np.float64)
    for c, f in enumerate(assignment):
        load[f] += demands[c]
    open_sum = np.sum(open_cost[load > 0])
    assign_sum = np.sum(assign_cost[np.arange(len(assignment)), assignment])
    return open_sum + assign_sum


def _valid(demands, capacities, assignment):
    if assignment is None:
        return False
    if len(assignment) != len(demands):
        return False
    load = np.zeros(len(capacities), dtype=np.float64)
    for c, f in enumerate(assignment):
        if f < 0 or f >= len(capacities):
            return False
        load[f] += demands[c]
    if np.any(load > capacities + 1e-9):
        return False
    return True


def _greedy_fallback(demands, capacities, assign_cost):
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(len(capacities), dtype=np.float64)
    for i in np.argsort(-demands):
        placed = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # put on the facility with most remaining capacity
            rem = capacities - load
            f = int(np.argmax(rem))
            load[f] += demands[i]
            result[i] = f
    return result


def _repair(demands, capacities, assignment, assign_cost):
    """Move over-capacity customers to facilities with spare room, cheapest first."""
    assignment = assignment.copy()
    load = np.zeros(len(capacities), dtype=np.float64)
    for c, f in enumerate(assignment):
        load[f] += demands[c]
    changed = True
    guard = 0
    while changed and guard < 20000:
        guard += 1
        changed = False
        # find overloaded facilities
        over = np.where(load > capacities + 1e-9)[0]
        if len(over) == 0:
            break
        # pick the most overloaded
        f_over = int(over[np.argmax(load[over] - capacities[over])])
        # pick a customer on it to move; prefer the one with best alternative
        custs = [c for c in range(len(assignment)) if assignment[c] == f_over]
        # sort by demand descending to free capacity fast
        custs.sort(key=lambda c: -demands[c])
        moved = False
        for c in custs:
            d = demands[c]
            # find a facility with room, cheapest
            cand = [f for f in range(len(capacities)) if load[f] + d <= capacities[f] + 1e-9]
            if not cand:
                continue
            cand.sort(key=lambda f: assign_cost[c, f])
            f_new = cand[0]
            load[f_over] -= d
            load[f_new] += d
            assignment[c] = f_new
            moved = True
            changed = True
            break
        if not moved:
            break
    return assignment


def _local_search(demands, capacities, open_cost, assign_cost, assignment, deadline):
    """Simulated Annealing optimization."""
    assignment = assignment.copy().astype(int)
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m, dtype=np.float64)
    for c, f in enumerate(assignment):
        load[f] += demands[c]
        
    def calc_cost():
        open_sum = np.sum(open_cost[load > 0])
        assign_sum = np.sum(assign_cost[np.arange(n), assignment])
        return open_sum + assign_cost_sum if False else open_sum + assign_sum # dummy line to avoid lint, actually just:
        
    # Correct cost calc
    def calc_cost():
        open_sum = np.sum(open_cost[load > 0])
        assign_sum = np.sum(assign_cost[np.arange(n), assignment])
        return open_sum + assign_sum
        
    current_cost = calc_cost()
    best_cost = current_cost
    best_assign = assignment.copy()
    
    # SA parameters
    start_temp = 1000.0
    end_temp = 1.0
    cooling_rate = 0.995
    temp = start_temp
    
    rng = np.random.default_rng(42)
    
    while temp > end_temp and time.time() < deadline:
        # Generate neighbor: pick random customer, move to random facility
        c = rng.integers(0, n)
        f_old = assignment[c]
        
        # Pick a random new facility
        f_new = rng.integers(0, m)
        if f_new == f_old:
            continue
            
        d = demands[c]
        
        # Check capacity
        if load[f_new] + d > capacities[f_new] + 1e-9:
            continue
            
        # Calculate delta
        # Old open cost contribution:
        # If load[f_old] == d, facility f_old becomes empty -> subtract open_cost[f_old]
        # If load[f_new] == 0, facility f_new becomes full -> add open_cost[f_new]
        open_delta = 0.0
        if load[f_old] == d:
            open_delta -= open_cost[f_old]
        if load[f_new] == 0:
            open_delta += open_cost[f_new]
            
        assign_delta = assign_cost[c, f_new] - assign_cost[c, f_old]
        total_delta = open_delta + assign_delta
        
        # Acceptance probability
        if total_delta < 0:
            accept = True
        else:
            # Avoid overflow/underflow issues
            if total_delta / temp > 700:
                accept = False
            else:
                accept = (rng.random() < np.exp(-total_delta / temp))
                
        if accept:
            # Apply move
            load[f_old] -= d
            load[f_new] += d
            assignment[c] = f_new
            current_cost += total_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assign = assignment.copy()
                
        temp *= cooling_rate
        
    return best_assign, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    n = len(demands)
    m = len(capacities)

    start = time.time()
    deadline = start + 120.0
    # safety margin
    safe_deadline = start + 110.0

    # fallback greedy
    fallback = _greedy_fallback(demands, capacities, assign_cost)
    if not _valid(demands, capacities, fallback):
        # last resort: assign all to capacity-sorted
        fallback = _greedy_fallback(demands, capacities, assign_cost)

    best_assign = fallback
    best_cost = _total_cost(demands, capacities, open_cost, assign_cost, fallback)

    # Penalty-based assignment with binary search on lambda
    # For penalty lambda, assign each customer to argmin(assign_cost[i,f] + lambda * open_cost[f]/capacities[f])
    # We want total load near total capacity.
    def penalty_assign(lam):
        # cost matrix with penalty
        penalty = lam * open_cost / np.maximum(capacities, 1e-9)
        total = assign_cost + penalty[None, :]
        return np.argmin(total, axis=1).astype(int)

    # search for lambda
    total_cap = np.sum(capacities)
    total_demand = np.sum(demands)
    lo = 0.0
    hi = 1e6
    best_lam = 0.0
    best_load_dev = float('inf')
    for _ in range(60):
        if time.time() > safe_deadline:
            break
        mid = (lo + hi) / 2
        a = penalty_assign(mid)
        load = np.zeros(m, dtype=np.float64)
        for c, f in enumerate(a):
            load[f] += demands[c]
        # deviation: we want load near total_cap but also feasible-ish
        # measure: sum of positive overload
        overload = np.sum(np.maximum(load - capacities, 0))
        underuse = np.sum(np.maximum(capacities - load, 0))
        dev = overload * 10 + underuse * 0.1
        if dev < best_load_dev:
            best_load_dev = dev
            best_lam = mid
            best_a = a
        if overload > 0:
            lo = mid
        else:
            hi = mid

    a = penalty_assign(best_lam)
    a = _repair(demands, capacities, a, assign_cost)
    if _valid(demands, capacities, a):
        a, c = _local_search(demands, capacities, open_cost, assign_cost, a, safe_deadline)
        if _valid(demands, capacities, a):
            cost = _total_cost(demands, capacities, open_cost, assign_cost, a)
            if cost < best_cost:
                best_cost = cost
                best_assign = a

    # also try greedy + local search
    if time.time() < safe_deadline:
        g = _greedy_fallback(demands, capacities, assign_cost)
        if _valid(demands, capacities, g):
            g, c = _local_search(demands, capacities, open_cost, assign_cost, g, safe_deadline)
            if _valid(demands, capacities, g):
                cost = _total_cost(demands, capacities, open_cost, assign_cost, g)
                if cost < best_cost:
                    best_cost = cost
                    best_assign = g

    # final safety check
    if not _valid(demands, capacities, best_assign):
        best_assign = fallback

    return best_assign.astype(int)
# EVOLVE-BLOCK-END
