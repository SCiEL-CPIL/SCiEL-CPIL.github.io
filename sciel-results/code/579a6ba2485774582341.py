# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

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

def _greedy_init(demands, capacities, open_cost, assign_cost, mode=0, rng=None):
    """Generate initial feasible assignment using greedy heuristic."""
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    if mode == 0:
        # Sort by demand descending
        order = np.argsort(-demands)
    elif mode == 1:
        # Random order
        order = rng.permutation(n)
    else:
        # Sort by minimum assignment cost ascending
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
    
    for i in order:
        d = demands[i]
        # Find feasible facilities
        feasible = np.where(load + d <= capacities)[0]
        
        if len(feasible) == 0:
            # Force assign to least loaded facility
            f = int(np.argmin(load))
            result[i] = f
            load[f] += d
            continue
        
        # Compute cost for each feasible facility
        costs = np.zeros(len(feasible))
        for idx, f in enumerate(feasible):
            costs[idx] = assign_cost[i, f]
            if load[f] == 0:
                costs[idx] += open_cost[f] * 0.5  # Penalty for opening new facility
        
        best_idx = np.argmin(costs)
        f = feasible[best_idx]
        result[i] = f
        load[f] += d
    
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit, rng=None):
    """Improve assignment via vectorized best-improvement moves."""
    n = len(demands)
    m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each customer to a better facility
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            
            cur_f = result[i]
            d = demands[i]
            
            # Find feasible alternative facilities
            feasible_mask = (load + d) <= capacities
            feasible_mask[cur_f] = False
            
            if not np.any(feasible_mask):
                continue
            
            feasible_indices = np.where(feasible_mask)[0]
            
            # Compute delta cost for moving to each feasible facility
            d_assign = assign_cost[i, feasible_indices] - assign_cost[i, cur_f]
            
            # Open cost delta: target opens if empty, source closes if becomes empty
            target_loads = load[feasible_indices]
            d_open_target = np.where(target_loads == 0, open_cost[feasible_indices], 0.0)
            
            # Source facility: will it become empty?
            if load[cur_f] - d == 0:
                d_open_source = -open_cost[cur_f]
            else:
                d_open_source = 0.0
            
            total_delta = d_assign + d_open_target + d_open_source
            
            min_idx = np.argmin(total_delta)
            min_delta = total_delta[min_idx]
            
            if min_delta < -1e-9:
                best_f = feasible_indices[min_idx]
                
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += min_delta
                improved = True
                
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_result = result.copy()
        
        # Try 2-opt swaps: swap two customers between facilities
        if not improved:
            break
    
    return best_result, best_cost

def _solve_cfl_lagrangian(demands, capacities, open_cost, assign_cost, time_limit):
    """Solve CFL using Lagrangian relaxation with subgradient method."""
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    
    # Dual variables (multipliers for capacity constraints)
    mu = np.zeros(m)
    
    best_upper = np.inf
    best_assignment = None
    
    def _repair_and_evaluate(mu):
        """Given dual multipliers, solve relaxed problem and repair to feasible solution."""
        relaxed_assignment = np.zeros(n, dtype=int)
        for i in range(n):
            scores = assign_cost[i, :] + mu
            relaxed_assignment[i] = np.argmin(scores)
        
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[relaxed_assignment[i]] += demands[i]
        
        # Move customers from over-capacity facilities
        for f in range(m):
            if load[f] > capacities[f] + 1e-9:
                custs_at_f = [i for i in range(n) if relaxed_assignment[i] == f]
                custs_at_f.sort(key=lambda i: -demands[i])
                
                excess = load[f] - capacities[f]
                for i in custs_at_f:
                    if excess <= 0:
                        break
                    
                    best_alt = -1
                    best_cost = np.inf
                    for alt_f in range(m):
                        if alt_f == f:
                            continue
                        if load[alt_f] + demands[i] <= capacities[alt_f]:
                            move_cost = assign_cost[i, alt_f] - assign_cost[i, f]
                            if load[alt_f] == 0:
                                move_cost += open_cost[alt_f]
                            if move_cost < best_cost:
                                best_cost = move_cost
                                best_alt = alt_f
                    
                    if best_alt >= 0:
                        load[f] -= demands[i]
                        load[best_alt] += demands[i]
                        relaxed_assignment[i] = best_alt
                        excess -= demands[i]
                    else:
                        min_load_f = np.argmin(load)
                        load[f] -= demands[i]
                        load[min_load_f] += demands[i]
                        relaxed_assignment[i] = min_load_f
        
        if not _is_valid(relaxed_assignment, demands, capacities):
            return None, np.inf
        
        cost = _total_cost(relaxed_assignment, demands, capacities, open_cost, assign_cost)
        return relaxed_assignment, cost
    
    # Subgradient method
    for iteration in range(300):
        if time.time() - start_time >= time_limit:
            break
        
        relaxed_assignment = np.zeros(n, dtype=int)
        for i in range(n):
            scores = assign_cost[i, :] + mu
            relaxed_assignment[i] = np.argmin(scores)
        
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[relaxed_assignment[i]] += demands[i]
        
        s = load - capacities
        
        step = 1.0 / (1 + 0.05 * iteration)
        mu = np.maximum(0, mu + step * s)
        
        assignment, cost = _repair_and_evaluate(mu)
        if assignment is not None and cost < best_upper:
            best_upper = cost
            best_assignment = assignment.copy()
        
        if iteration > 50 and best_upper == np.inf:
            break
    
    if best_assignment is not None:
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.1:
            best_assignment, _ = _local_search(
                best_assignment, demands, capacities, open_cost, assign_cost, 
                np.random.RandomState(42), time_remaining
            )
    
    return best_assignment, best_upper

def _solve_cfl_lp(demands, capacities, open_cost, assign_cost, time_limit):
    """Try to solve CFL as a linear program (may be too slow for large instances)."""
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    
    # Variables: x[i,f] = 1 if customer i assigned to facility f
    #            y[f] = 1 if facility f is open
    # Total variables: n*m + m
    
    # Objective: minimize sum_i,f assign_cost[i,f]*x[i,f] + sum_f open_cost[f]*y[f]
    c = np.concatenate([assign_cost.flatten(), open_cost])
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1 for all i
    # 2. Capacity: sum_i demands[i]*x[i,f] <= capacities[f] for all f
    # 3. Facility opening: x[i,f] <= y[f] for all i,f
    
    # Build constraint matrix
    num_constraints = n + m + n*m
    A_ub = np.zeros((num_constraints, n*m + m))
    b_ub = np.zeros(num_constraints)
    
    # Constraint 1: sum_f x[i,f] = 1 (use equality, so we handle separately)
    # Actually, let's use equality constraints
    A_eq = np.zeros((n, n*m + m))
    b_eq = np.ones(n)
    
    for i in range(n):
        for f in range(m):
            A_eq[i, i*m + f] = 1.0
    
    # Constraint 2: Capacity constraints
    for f in range(m):
        row = m + f
        for i in range(n):
            A_ub[row, i*m + f] = demands[i]
        b_ub[row] = capacities[f]
    
    # Constraint 3: x[i,f] <= y[f]
    for i in range(n):
        for f in range(m):
            row = m + m + i*m + f
            A_ub[row, i*m + f] = 1.0
            A_ub[row, n*m + f] = -1.0
            b_ub[row] = 0.0
    
    # Bounds: x[i,f] in [0,1], y[f] in [0,1]
    bounds = [(0, 1)] * (n*m + m)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.success:
            # Extract assignment
            assignment = np.zeros(n, dtype=int)
            x = result.x[:n*m].reshape(n, m)
            for i in range(n):
                assignment[i] = np.argmax(x[i])
            
            # Check validity
            if _is_valid(assignment, demands, capacities):
                cost = _total_cost(assignment, demands, capacities, open_cost, assign_cost)
                return assignment, cost
    except Exception as e:
        pass
    
    return None, np.inf

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    global_start = time.time()
    deadline = global_start + 100.0
    
    rng = np.random.RandomState(42)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: LP solver (try first, may give good solution)
    lp_time = 20.0
    lp_result, lp_cost = _solve_cfl_lp(demands, capacities, open_cost, assign_cost, lp_time)
    
    if lp_result is not None and lp_cost < best_cost:
        best_cost = lp_cost
        best_result = lp_result.copy()
    
    # Phase 2: Lagrangian relaxation
    lagr_time = 30.0
    lagr_result, lagr_cost = _solve_cfl_lagrangian(demands, capacities, open_cost, assign_cost, lagr_time)
    
    if lagr_result is not None and lagr_cost < best_cost:
        best_cost = lagr_cost
        best_result = lagr_result.copy()
    
    # Phase 3: Multi-start greedy + local search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        for restart in range(8):
            if time.time() >= deadline:
                break
            
            mode = restart % 3
            seed_result = _greedy_init(demands, capacities, open_cost, assign_cost, mode=mode, rng=rng)
            
            if not _is_valid(seed_result, demands, capacities):
                continue
            
            seed_cost = _total_cost(seed_result, demands, capacities, open_cost, assign_cost)
            
            if seed_cost < best_cost:
                best_cost = seed_cost
                best_result = seed_result.copy()
            
            time_remaining = deadline - time.time()
            ls_time = min(5.0, time_remaining * 0.3)
            if ls_time <= 0:
                break
            
            ls_result, ls_cost = _local_search(
                seed_result, demands, capacities, open_cost, assign_cost, rng, ls_time
            )
            
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_result = ls_result.copy()
    
    # Fallback
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, mode=0, rng=rng)
    
    if not _is_valid(best_result, demands, capacities):
        load = np.zeros(m, dtype=float)
        best_result = np.zeros(n, dtype=int)
        for i in np.argsort(-demands):
            feasible = [f for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if feasible:
                f = min(feasible, key=lambda x: assign_cost[i, x])
            else:
                f = int(np.argmin(load))
            best_result[i] = f
            load[f] += demands[i]
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
