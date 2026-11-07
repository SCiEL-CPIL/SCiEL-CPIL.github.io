"""Capacitated facility location: MILP via HiGHS with greedy warm-start UB."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

# We'll use scipy.optimize.milp if available, otherwise fall back to greedy
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except ImportError:
    _HAS_MILP = False


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Fast greedy: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    # Sort customers by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Get facilities sorted by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            f = np.argmax(rem)
            result[i] = f
            load[f] += demands[i]
    return result


def _compute_cost(demands, capacities, open_cost, assign_cost, assignment):
    """Compute total cost of an assignment."""
    n_fac = len(capacities)
    # Open cost: sum of open_cost for facilities with at least one customer
    used = np.zeros(n_fac, dtype=bool)
    for c in assignment:
        used[c] = True
    total = np.sum(open_cost[used])
    # Assignment cost
    for i in range(len(demands)):
        total += assign_cost[i, assignment[i]]
    return total


def _milp_solve(demands, capacities, open_cost, assign_cost, ub):
    """
    Solve capacitated facility location via MILP using scipy.optimize.milp.
    
    Variables:
      x[i, f] in {0,1}: customer i assigned to facility f
      y[f] in {0,1}: facility f is open
    
    Minimize: sum_f open_cost[f]*y[f] + sum_{i,f} assign_cost[i,f]*x[i,f]
    
    Subject to:
      sum_f x[i,f] = 1 for all i
      sum_i demands[i]*x[i,f] <= capacities[f]*y[f] for all f
      x[i,f] = 0 if y[f] = 0 (enforced by capacity constraint when demands > 0)
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Total number of variables: n_cust * n_fac (x) + n_fac (y)
    # x is indexed 0..n_cust*n_fac-1, y is indexed n_cust*n_fac..n_cust*n_fac+n_fac-1
    n_x = n_cust * n_fac
    n_vars = n_x + n_fac
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_x + f] = open_cost[f]
    
    # Integrality: all binary
    integrality = np.ones(n_vars, dtype=int)
    
    # Bounds: all in [0, 1]
    lb = np.zeros(n_vars)
    ub_arr = np.ones(n_vars)
    bounds = Bounds(lb, ub_arr)
    
    # Constraints
    # 1) Each customer assigned to exactly one facility: sum_f x[i,f] = 1
    # 2) Capacity: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
    
    # Build A matrix and b for linear constraints
    # We'll use LinearConstraint with separate rows
    
    # Constraint rows:
    # For each customer i: sum_f x[i,f] = 1  ->  1 row per customer (equality)
    # For each facility f: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0  ->  1 row per facility (inequality)
    
    n_cons = n_cust + n_fac
    
    # A matrix: n_cons x n_vars
    A = np.zeros((n_cons, n_vars))
    
    # Customer assignment constraints (rows 0..n_cust-1)
    for i in range(n_cust):
        for f in range(n_fac):
            A[i, i * n_fac + f] = 1.0
    
    # Capacity constraints (rows n_cust..n_cust+n_fac-1)
    for f in range(n_fac):
        row = n_cust + f
        for i in range(n_cust):
            A[row, i * n_fac + f] = demands[i]
        A[row, n_x + f] = -capacities[f]
    
    # Lower and upper bounds for constraints
    cons_lb = np.zeros(n_cons)
    cons_ub = np.zeros(n_cons)
    
    # Customer constraints: equality -> lb = ub = 1
    cons_lb[:n_cust] = 1.0
    cons_ub[:n_cust] = 1.0
    
    # Capacity constraints: <= 0 -> lb = -inf, ub = 0
    cons_lb[n_cust:] = -np.inf
    cons_ub[n_cons] = 0.0  # This is wrong, fix below
    
    # Actually set properly
    cons_lb = np.concatenate([np.ones(n_cust), np.full(n_fac, -np.inf)])
    cons_ub = np.concatenate([np.ones(n_cust), np.zeros(n_fac)])
    
    constraints = LinearConstraint(A, cons_lb, cons_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 100, "mip_rel_gap": 0.0, "mip_abs_gap": 0.0})
        
        if res.success or res.status == 1:  # 1 = optimal found
            x_flat = res.x[:n_x]
            x = x_flat.reshape(n_cust, n_fac)
            assignment = np.argmax(x, axis=1).astype(int)
            return assignment
        elif res.x is not None:
            # Use best feasible solution found
            x_flat = res.x[:n_x]
            x = x_flat.reshape(n_cust, n_fac)
            assignment = np.argmax(x, axis=1).astype(int)
            return assignment
    except Exception:
        pass
    
    return None


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility, respecting capacities.
    Returns integer array of facility indices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Step 1: Get a good initial solution via greedy
    greedy_sol = _greedy_assign(demands, capacities, open_cost, assign_cost)
    greedy_cost = _compute_cost(demands, capacities, open_cost, assign_cost, greedy_sol)
    
    best_sol = greedy_sol.copy()
    best_cost = greedy_cost
    
    # Step 2: Try MILP if available and time permits
    if _HAS_MILP and (deadline - time.time()) > 5.0:
        milp_sol = _milp_solve(demands, capacities, open_cost, assign_cost, greedy_cost)
        if milp_sol is not None:
            milp_cost = _compute_cost(demands, capacities, open_cost, assign_cost, milp_sol)
            if milp_cost < best_cost:
                best_sol = milp_sol
                best_cost = milp_cost
    
    # Step 3: Local search improvement if time remains
    if (deadline - time.time()) > 3.0:
        best_sol = _local_search(demands, capacities, open_cost, assign_cost, best_sol, best_cost, deadline)
    
    # Validate and return
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        f = best_sol[i]
        if f < 0 or f >= n_fac:
            f = 0
        result[i] = f
        load[f] += demands[i]
    
    # Final capacity check - if violated, reassign
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            # Move some customers from this facility
            overflow = load[f] - capacities[f]
            custs_at_f = [i for i in range(n_cust) if result[i] == f]
            # Sort by demand descending, move largest first
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if overflow <= 0:
                    break
                # Find alternative facility
                for alt in np.argsort(assign_cost[i]):
                    if alt != f and load[alt] + demands[i] <= capacities[alt]:
                        load[f] -= demands[i]
                        load[alt] += demands[i]
                        result[i] = alt
                        overflow -= demands[i]
                        break
    
    return result.astype(int)


def _local_search(demands, capacities, open_cost, assign_cost, init_sol, init_cost, deadline):
    """Simple local search: try moving each customer to a better facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    sol = init_sol.copy()
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[sol[i]] += demands[i]
    
    current_cost = init_cost
    improved = True
    max_iters = 50
    
    while improved and (deadline - time.time()) > 2.0:
        improved = False
        for _ in range(max_iters):
            if (deadline - time.time()) < 2.0:
                break
            for i in range(n_cust):
                if (deadline - time.time()) < 2.0:
                    break
                cur_f = sol[i]
                # Try all other facilities
                for alt_f in range(n_fac):
                    if alt_f == cur_f:
                        continue
                    # Check capacity
                    if load[alt_f] + demands[i] <= capacities[alt_f]:
                        # Compute cost change
                        # Remove i from cur_f, add to alt_f
                        cost_change = assign_cost[i, alt_f] - assign_cost[i, cur_f]
                        
                        # Check if cur_f becomes empty (save open cost)
                        cur_count = np.sum(sol == cur_f)
                        alt_count = np.sum(sol == alt_f)
                        
                        if cur_count == 1 and alt_count == 0:
                            cost_change += open_cost[alt_f] - open_cost[cur_f]
                        elif cur_count == 1 and alt_count > 0:
                            cost_change -= open_cost[cur_f]
                        elif cur_count > 1 and alt_count == 0:
                            cost_change += open_cost[alt_f]
                        
                        if cost_change < -1e-9:
                            # Apply move
                            load[cur_f] -= demands[i]
                            load[alt_f] += demands[i]
                            sol[i] = alt_f
                            current_cost += cost_change
                            improved = True
                            break  # Move to next customer
    
    return sol
# EVOLVE-BLOCK-END
