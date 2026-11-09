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
    """Generate an initial feasible assignment."""
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
        feasible = []
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                if mode == 2:
                    penalty = open_cost[f] if load[f] == 0 else 0.0
                    feasible.append((assign_cost[i, f] + penalty * 0.1, f))
                else:
                    feasible.append((assign_cost[i, f], f))
        if not feasible:
            feasible = [(assign_cost[i, f], f) for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
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
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            
            cur_f = result[i]
            d = demands[i]
            
            feasible_mask = (load + d) <= capacities
            feasible_mask[cur_f] = False
            
            if not np.any(feasible_mask):
                continue
            
            feasible_indices = np.where(feasible_mask)[0]
            target_facs = feasible_indices
            target_loads = load[target_facs]
            
            d_assign = assign_cost[i, target_facs] - assign_cost[i, cur_f]
            d_open_target = np.where(target_loads == 0, open_cost[target_facs], 0.0)
            d_open_source = -open_cost[cur_f] if (load[cur_f] - d) == 0 else 0.0
            
            total_delta = d_assign + d_open_target + d_open_source
            
            min_idx = np.argmin(total_delta)
            min_delta = total_delta[min_idx]
            
            if min_delta < -1e-9:
                best_f = target_facs[min_idx]
                
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += min_delta
                improved = True
                
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_result = result.copy()
        
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
    
    # Initial step size
    step = 1.0
    best_upper = np.inf
    best_assignment = None
    best_dual = np.zeros(m)
    
    # Upper bound from a heuristic repair
    def _repair_and_evaluate(mu):
        """Given dual multipliers, solve relaxed problem and repair to feasible solution."""
        # Relaxed problem: minimize sum_i (assign_cost[i,f] + mu[f]) + sum_f open_cost[f]*y[f]
        # subject to y[f] >= 1 if any customer assigned to f
        # For each customer, choose facility minimizing (assign_cost[i,f] + mu[f])
        # But we need to track which facilities are "opened"
        
        # Simple approach: for each customer, pick facility with min (assign_cost[i,f] + mu[f])
        # Then repair capacity violations by moving customers to other facilities
        
        relaxed_assignment = np.zeros(n, dtype=int)
        for i in range(n):
            # Minimize assign_cost[i,f] + mu[f]
            scores = assign_cost[i, :] + mu
            relaxed_assignment[i] = np.argmin(scores)
        
        # Now repair capacity violations
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[relaxed_assignment[i]] += demands[i]
        
        # Move customers from over-capacity facilities
        # Sort customers by how much they "want" to stay (lower mu penalty)
        # Simple repair: for over-capacity facilities, move customers to cheapest feasible alternative
        for f in range(m):
            if load[f] > capacities[f] + 1e-9:
                # Get customers at f, sorted by demand (largest first to move)
                custs_at_f = [i for i in range(n) if relaxed_assignment[i] == f]
                custs_at_f.sort(key=lambda i: -demands[i])
                
                excess = load[f] - capacities[f]
                for i in custs_at_f:
                    if excess <= 0:
                        break
                    # Find best alternative facility
                    best_alt = -1
                    best_cost = np.inf
                    for alt_f in range(m):
                        if alt_f == f:
                            continue
                        if load[alt_f] + demands[i] <= capacities[alt_f]:
                            # Cost to move: assign_cost[i, alt_f] - assign_cost[i, f]
                            # Plus potential open cost if alt_f was empty
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
                        # Cannot repair - force assign to least loaded
                        min_load_f = np.argmin(load)
                        load[f] -= demands[i]
                        load[min_load_f] += demands[i]
                        relaxed_assignment[i] = min_load_f
        
        # Check validity
        if not _is_valid(relaxed_assignment, demands, capacities):
            return None, np.inf
        
        cost = _total_cost(relaxed_assignment, demands, capacities, open_cost, assign_cost)
        return relaxed_assignment, cost
    
    # Subgradient method
    for iteration in range(500):
        if time.time() - start_time >= time_limit:
            break
        
        # Solve relaxed problem: for each customer, choose facility minimizing (assign_cost[i,f] + mu[f])
        # This gives a lower bound
        relaxed_assignment = np.zeros(n, dtype=int)
        for i in range(n):
            scores = assign_cost[i, :] + mu
            relaxed_assignment[i] = np.argmin(scores)
        
        # Compute load
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[relaxed_assignment[i]] += demands[i]
        
        # Subgradient: s_f = load[f] - capacities[f]
        s = load - capacities
        
        # Lower bound
        lb = np.sum(assign_cost[np.arange(n), relaxed_assignment] + mu[relaxed_assignment])
        # Note: This is not quite right for the lower bound with open costs, but it's a valid LB for the relaxed problem
        
        # Update multipliers
        # Step size: use Polyak-style or diminishing
        # Simple diminishing: step = step / (1 + 0.1 * iteration)
        step = 1.0 / (1 + 0.05 * iteration)
        
        # Update mu: mu_f = max(0, mu_f + step * s_f)
        mu = np.maximum(0, mu + step * s)
        
        # Try to repair and get upper bound
        assignment, cost = _repair_and_evaluate(mu)
        if assignment is not None and cost < best_upper:
            best_upper = cost
            best_assignment = assignment.copy()
            best_dual = mu.copy()
        
        # Early stopping if no improvement in last 50 iterations
        if iteration > 50 and best_upper == np.inf:
            break
    
    # Final local search on best assignment
    if best_assignment is not None:
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.1:
            best_assignment, _ = _local_search(
                best_assignment, demands, capacities, open_cost, assign_cost, np.random.RandomState(42), time_remaining
            )
    
    return best_assignment, best_upper

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    # Deadline guard
    global_start = time.time()
    deadline = global_start + 100.0
    
    rng = np.random.RandomState(42)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Lagrangian relaxation
    lagr_time = 30.0
    lagr_result, lagr_cost = _solve_cfl_lagrangian(demands, capacities, open_cost, assign_cost, lagr_time)
    
    if lagr_result is not None and lagr_cost < best_cost:
        best_cost = lagr_cost
        best_result = lagr_result.copy()
    
    # Phase 2: Multi-start greedy + local search with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        for restart in range(10):
            if time.time() >= deadline:
                break
            
            mode = restart % 3
            seed_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=mode)
            
            if not _is_valid(seed_result, demands, capacities):
                continue
            
            seed_cost = _total_cost(seed_result, demands, capacities, open_cost, assign_cost)
            
            if seed_cost < best_cost:
                best_cost = seed_cost
                best_result = seed_result.copy()
            
            time_remaining = deadline - time.time()
            ls_time = min(3.0, time_remaining * 0.3)
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
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=0)
    
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
