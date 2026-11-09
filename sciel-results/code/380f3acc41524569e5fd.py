"""Capacitated facility location: MILP via HiGHS with greedy fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _greedy(demands, capacities, open_cost, assign_cost, order=None):
    """Greedy largest-first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    if order is None:
        order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        # facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + d <= capacities[f]:
                load[f] += d
                result[i] = f
                placed = True
                break
        if not placed:
            # fallback: facility with most remaining capacity
            rem = capacities - load
            f = int(np.argmax(rem))
            load[f] += d
            result[i] = f
    return result


def _total_cost(demands, capacities, open_cost, assign_cost, assignment):
    """Compute the objective value for a given assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if assignment is None:
        return np.inf
    assignment = np.asarray(assignment, dtype=int)
    if len(assignment) != n_cust:
        return np.inf
    if np.any(assignment < 0) or np.any(assignment >= n_fac):
        return np.inf
    # capacity check
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    if np.any(load > capacities + 1e-6):
        return np.inf
    # cost
    used = np.zeros(n_fac, dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = float(np.sum(open_cost[used]))
    assign_sum = float(np.sum(assign_cost[np.arange(n_cust), assignment]))
    return open_sum + assign_sum


def _milp_solve(demands, capacities, open_cost, assign_cost, time_limit):
    """Solve the capacitated facility location MILP. Returns assignment or None."""
    n_cust = len(demands)
    n_fac = len(capacities)
    # Variables: x[i, f] for i in [0, n_cust), f in [0, n_fac); y[f] for f in [0, n_fac)
    n_vars = n_cust * n_fac + n_fac
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]

    integrality = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            integrality[i * n_fac + f] = 1
    for f in range(n_fac):
        integrality[n_cust * n_fac + f] = 1

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)

    # A: rows
    # For each customer i: sum_f x[i,f] = 1
    # For each facility f: sum_i x[i,f] - y[f] * cap[f] <= 0
    n_rows = n_cust + n_fac
    A = np.zeros((n_rows, n_vars))
    for i in range(n_cust):
        for f in range(n_fac):
            A[i, i * n_fac + f] = 1.0
    for f in range(n_fac):
        for i in range(n_cust):
            A[n_cust + f, i * n_fac + f] = 1.0
        A[n_cust + f, n_cust * n_fac + f] = -capacities[f]

    row_lb = np.zeros(n_rows)
    row_ub = np.zeros(n_rows)
    for i in range(n_cust):
        row_lb[i] = 1.0
        row_ub[i] = 1.0
    for f in range(n_fac):
        row_lb[n_cust + f] = -np.inf
        row_ub[n_cust + f] = 0.0

    constraints = LinearConstraint(A, lb=row_lb, ub=row_ub)
    bounds = Bounds(lb=lb, ub=ub)

    t_start = time.time()
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": time_limit, "presolve": True})
    except Exception:
        return None

    if res is None or not res.success or res.x is None:
        return None

    x = res.x
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        row = x[i * n_fac:(i + 1) * n_fac]
        # pick the facility with max x (should be ~1)
        assignment[i] = int(np.argmax(row))
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Deadline guard: 150s wall-clock, use 135s as safety margin
    t0 = time.time()
    deadline = t0 + 135.0

    # Greedy baseline
    best_assignment = _greedy(demands, capacities, open_cost, assign_cost)
    best_cost = _total_cost(demands, capacities, open_cost, assign_cost, best_assignment)

    # Try MILP
    remaining = deadline - time.time()
    if remaining > 2.0:
        milp_time = max(1.0, min(remaining - 2.0, 120.0))
        milp_assignment = _milp_solve(demands, capacities, open_cost, assign_cost, milp_time)
        if milp_assignment is not None:
            milp_cost = _total_cost(demands, capacities, open_cost, assign_cost, milp_assignment)
            if milp_cost < best_cost:
                best_assignment = milp_assignment
                best_cost = milp_cost

    # If MILP didn't help, try a few greedy variants with different orderings
    remaining = deadline - time.time()
    if remaining > 1.0:
        # Try ordering by demand ascending, and by random
        variants = []
        # ascending demand
        variants.append(np.argsort(demands))
        # random order
        rng = np.random.default_rng(42)
        variants.append(rng.permutation(n_cust))
        for order in variants:
            if time.time() > deadline - 0.5:
                break
            a = _greedy(demands, capacities, open_cost, assign_cost, order=order)
            c = _total_cost(demands, capacities, open_cost, assign_cost, a)
            if c < best_cost:
                best_assignment = a
                best_cost = c

    # Final safety: ensure validity
    final = _total_cost(demands, capacities, open_cost, assign_cost, best_assignment)
    if np.isinf(final):
        # last resort: pure greedy
        best_assignment = _greedy(demands, capacities, open_cost, assign_cost)

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
