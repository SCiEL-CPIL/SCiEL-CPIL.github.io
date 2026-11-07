# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

def _compute_cost(result, demands, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    if result is None or len(result) != len(demands):
        return np.inf
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        if 0 <= f < len(open_cost):
            open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open

def _is_valid(result, demands, capacities):
    """Check if assignment is valid."""
    if result is None or len(result) != len(demands):
        return False
    load = np.zeros(len(capacities))
    for i in range(len(demands)):
        f = result[i]
        if f < 0 or f >= len(capacities):
            return False
        load[f] += demands[i]
    return np.all(load <= capacities + 1e-6)

def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(100):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            # Find customers at f_over, move those with most savings to other facilities
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            # Sort by demand descending to clear excess faster
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                # Find cheapest feasible facility (not f_over)
                best_f = -1
                best_cost = np.inf
                for f in range(m):
                    if f == f_over:
                        continue
                    if load[f] + d <= capacities[f] + 1e-6:
                        c = assign_cost[i, f]
                        if c < best_cost:
                            best_cost = c
                            best_f = f
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    # Fallback: move to facility with most remaining capacity
                    rem = capacities - load
                    rem[f_over] = -np.inf # Don't move back to overloaded
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=50, deadline=None, rng=None):
    """Local search with single moves."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n)
        
        # Single moves
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if not improved:
            break
    
    return result, current_cost

def _solve_lp(demands, capacities, open_cost, assign_cost):
    """Solve the LP relaxation of the CFLP."""
    n = len(demands)
    m = len(capacities)
    
    # Variables: x_ij (n*m) + y_j (m)
    num_vars = n * m + m
    
    # Objective
    c = np.zeros(num_vars)
    for i in range(n):
        for j in range(m):
            c[i*m + j] = assign_cost[i, j]
    for j in range(m):
        c[n*m + j] = open_cost[j]
        
    # Constraints
    # 1. Assignment: sum_j x_ij = 1
    # 2. Capacity: sum_i d_i x_ij <= C_j y_j  => sum_i d_i x_ij - C_j y_j <= 0
    # 3. Linking: x_ij <= y_j => x_ij - y_j <= 0
    
    A_rows = []
    lb = []
    ub = []
    
    # Assignment constraints
    for i in range(n):
        row = np.zeros(num_vars)
        for j in range(m):
            row[i*m + j] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
        
    # Capacity constraints
    for j in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            row[i*m + j] = demands[i]
        row[n*m + j] = -capacities[j]
        A_rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        
    # Linking constraints
    for i in range(n):
        for j in range(m):
            row = np.zeros(num_vars)
            row[i*m + j] = 1.0
            row[n*m + j] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            
    A = np.array(A_rows)
    
    bounds = [(0, 1) for _ in range(num_vars)]
    
    res = linprog(c, A_ub=A, b_ub=np.array(ub), bounds=bounds, method='highs')
    
    if res.status == 0:
        x_flat = res.x[:n*m].reshape(n, m)
        y = res.x[n*m:]
        return x_flat, y
    return None, None

def _heuristic_init_from_lp(lp_x, lp_y, demands, capacities, assign_cost):
    """Create an integer solution from LP solution."""
    n = len(demands)
    m = len(capacities)
    
    # Determine open facilities
    # If y_j > 0.5, open it.
    open_facilities = np.where(lp_y > 0.5)[0]
    
    # If no facilities are open (shouldn't happen with valid LP), open the one with lowest open cost
    if len(open_facilities) == 0:
        open_facilities = np.array([np.argmin(open_cost)])
        
    # Ensure we have enough capacity. If sum of capacities of open < total demand, open more.
    total_demand = np.sum(demands)
    current_cap = np.sum(capacities[open_facilities])
    if current_cap < total_demand:
        # Open facilities with lowest open cost until capacity is sufficient
        closed = np.setdiff1d(np.arange(m), open_facilities)
        closed = closed[np.argsort(open_cost[closed])]
        for f in closed:
            if current_cap >= total_demand:
                break
            open_facilities = np.append(open_facilities, f)
            current_cap += capacities[f]
            
    # Assign customers
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    
    # Sort customers by demand descending (largest first is usually better for packing)
    order = np.argsort(-demands)
    
    for i in order:
        d = demands[i]
        best_f = -1
        best_c = np.inf
        
        # Try open facilities first, sorted by cost
        # We can just iterate all open facilities
        for f in open_facilities:
            if load[f] + d <= capacities[f] + 1e-6:
                c = assign_cost[i, f]
                if c < best_c:
                    best_c = c
                    best_f = f
                    
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            # If no open facility has space, assign to the one with most remaining capacity
            # This might violate capacity if total cap < total demand, but we ensured cap >= demand above
            # However, greedy might fill them up unevenly.
            rem = capacities - load
            rem[~np.isin(np.arange(m), open_facilities)] = -np.inf # Prefer open ones
            # If all open are full, we might need to open a new one or repair later
            best_f = int(np.argmax(rem))
            result[i] = best_f
            load[best_f] += d
            
    return result

def _solve_mip(demands, capacities, open_cost, assign_cost, deadline, initial_solution=None):
    """Solve the CFLP using MIP."""
    n = len(demands)
    m = len(capacities)
    
    num_vars = n * m + m
    
    c = np.zeros(num_vars)
    for i in range(n):
        for j in range(m):
            c[i*m + j] = assign_cost[i, j]
    for j in range(m):
        c[n*m + j] = open_cost[j]
        
    A_rows = []
    lb = []
    ub = []
    
    # Assignment
    for i in range(n):
        row = np.zeros(num_vars)
        for j in range(m):
            row[i*m + j] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
        
    # Capacity
    for j in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            row[i*m + j] = demands[i]
        row[n*m + j] = -capacities[j]
        A_rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        
    # Linking
    for i in range(n):
        for j in range(m):
            row = np.zeros(num_vars)
            row[i*m + j] = 1.0
            row[n*m + j] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            
    A = np.array(A_rows)
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    integrality = np.ones(num_vars)
    
    time_limit = deadline - time.time() - 2.0
    if time_limit <= 0:
        return None
        
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality, options={"time_limit": time_limit})
        
        if res.status == 0 and res.x is not None:
            x = res.x[:n*m].reshape(n, m)
            result = np.argmax(x, axis=1).astype(int)
            
            if _is_valid(result, demands, capacities):
                return result
            else:
                result = _repair(result, demands, capacities, assign_cost)
                return result
        else:
            return None
    except Exception:
        return None

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = np.inf
    
    # 1. LP Relaxation
    lp_x, lp_y = _solve_lp(demands, capacities, open_cost, assign_cost)
    
    if lp_x is not None:
        # 2. Heuristic Init from LP
        result = _heuristic_init_from_lp(lp_x, lp_y, demands, capacities, assign_cost)
        result = _repair(result, demands, capacities, assign_cost)
        
        # 3. Local Search
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=20, deadline=deadline, rng=rng)
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    else:
        # Fallback to Greedy if LP fails
        result = np.full(n, -1, dtype=int)
        load = np.zeros(m)
        for i in np.argsort(-demands):
            d = demands[i]
            best_f = -1
            best_c = np.inf
            for f in range(m):
                if load[f] + d <= capacities[f] + 1e-6:
                    c = assign_cost[i, f]
                    if c < best_c:
                        best_c = c
                        best_f = f
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += d
            else:
                rem = capacities - load
                best_f = int(np.argmax(rem))
                result[i] = best_f
                load[best_f] += d
                
        result = _repair(result, demands, capacities, assign_cost)
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=20, deadline=deadline, rng=rng)
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
            
    # 4. MIP Polish if time permits
    if time.time() < deadline - 5:
        mip_result = _solve_mip(demands, capacities, open_cost, assign_cost, deadline)
        if mip_result is not None:
            mip_cost = _compute_cost(mip_result, demands, open_cost, assign_cost)
            if mip_cost < best_cost:
                best_cost = mip_cost
                best_result = mip_result.copy()
                
    # Final validity check
    if best_result is not None:
        if not _is_valid(best_result, demands, capacities):
            best_result = _repair(best_result, demands, capacities, assign_cost)
            
    if best_result is None:
        best_result = np.zeros(n, dtype=int)
        best_result = _repair(best_result, demands, capacities, assign_cost)
        
    return best_result
# EVOLVE-BLOCK-END
