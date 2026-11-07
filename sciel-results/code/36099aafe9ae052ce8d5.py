# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MIP with Lagrangian relaxation fallback
    and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 145s safety margin out of 150s

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Try MIP first
    assignment = None
    mip_elapsed = 0.0
    try:
        assignment, mip_elapsed = _solve_mip(demands, capacities, open_cost, assign_cost, deadline)
    except Exception:
        pass

    # If MIP didn't produce a valid solution, use Lagrangian relaxation + repair
    if assignment is None or not _is_valid(assignment, demands, capacities):
        assignment = _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, deadline)

    # Local search polish
    if time.time() < deadline:
        assignment = _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline)

    # Final validation
    if not _is_valid(assignment, demands, capacities):
        assignment = _greedy_fallback(demands, capacities, open_cost, assign_cost)

    return assignment.astype(int)


def _solve_mip(demands, capacities, open_cost, assign_cost, deadline):
    """Solve CFLP using Mixed Integer Programming."""
    n_cust = len(demands)
    n_fac = len(capacities)
    n_vars = n_cust * n_fac + n_fac

    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]

    # Constraints
    n_constraints = n_cust + n_fac + n_cust * n_fac
    A = np.zeros((n_constraints, n_vars))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)

    row = 0
    # Each customer assigned to exactly one facility
    for i in range(n_cust):
        for f in range(n_fac):
            A[row, i * n_fac + f] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1

    # Capacity constraint: sum_i demands[i] * x[i,f] <= capacities[f] * y[f]
    for f in range(n_fac):
        for i in range(n_cust):
            A[row, i * n_fac + f] = demands[i]
        A[row, n_cust * n_fac + f] = -capacities[f]
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1

    # Linking: x[i,f] <= y[f]
    for i in range(n_cust):
        for f in range(n_fac):
            A[row, i * n_fac + f] = 1.0
            A[row, n_cust * n_fac + f] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1

    integrality = np.ones(n_vars)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    constraint = LinearConstraint(A, lb, ub)

    # Time remaining for MIP
    elapsed = time.time() - (deadline - 145.0)  # approximate
    time_limit = max(1.0, 140.0 - elapsed)

    result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds,
                  options={"time_limit": time_limit})

    elapsed_mip = time.time() - (deadline - 145.0)

    if not result.success or result.x is None:
        return None, elapsed_mip

    x = result.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        assignment[i] = np.argmax(x[i])

    return assignment, elapsed_mip


def _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, deadline):
    """
    Lagrangian relaxation on capacity constraints with subgradient optimization.
    Then repair capacity violations.
    """
    n_cust = len(demands)
    n_fac = len(capacities)

    # Dual variables for capacity constraints
    lambda_f = np.zeros(n_fac)

    # Allocate opening cost per unit capacity for the Lagrangian
    # We'll use a simple allocation: open_cost[f] / capacities[f] per unit demand
    unit_open_cost = np.zeros(n_fac)
    for f in range(n_fac):
        if capacities[f] > 0:
            unit_open_cost[f] = open_cost[f] / capacities[f]

    best_assignment = None
    best_cost = np.inf

    # Subgradient optimization
    max_iters = 500
    step_size_factor = 2.0
    last_best_iter = 0

    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break

        # For each customer, choose facility minimizing assign_cost[i,f] + lambda_f + unit_open_cost[f] * demands[i]
        # The Lagrangian relaxed problem decomposes by customer
        # L(λ) = sum_i min_f (assign_cost[i,f] + lambda_f * demands[i]) + sum_f (lambda_f * capacities[f] - open_cost[f])
        # But we need to handle the opening cost properly

        # Simpler approach: for each customer, effective cost is assign_cost[i,f] + lambda_f * demands[i]
        # Then add back the opening cost for used facilities
        effective_cost = assign_cost + np.outer(demands, lambda_f)
        assignment = np.argmin(effective_cost, axis=1)

        # Compute load
        load = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]

        # Compute actual cost
        used_facilities = set(assignment.tolist())
        total_cost = 0.0
        for i in range(n_cust):
            total_cost += assign_cost[i, assignment[i]]
        for f in used_facilities:
            total_cost += open_cost[f]

        # Check feasibility
        feasible = all(load[f] <= capacities[f] for f in range(n_fac))

        if feasible and total_cost < best_cost:
            best_cost = total_cost
            best_assignment = assignment.copy()
            last_best_iter = it

        # Subgradient update
        # g_f = load[f] - capacities[f] (violation)
        g = load - capacities.astype(np.float64)

        # Step size
        # Upper bound on optimal: use a simple feasible solution cost
        if best_assignment is not None:
            ub_val = best_cost
        else:
            # Use greedy as upper bound
            greedy = _greedy_fallback(demands, capacities, open_cost, assign_cost)
            ub_val = _compute_cost(greedy, demands, capacities, open_cost, assign_cost)

        lb_val = np.sum(np.min(assign_cost + np.outer(demands, lambda_f), axis=1)) + \
                 np.sum(lambda_f * capacities) - np.sum(open_cost)
        # Actually, let's use a simpler step size
        # step = step_size_factor * (ub_val - lb_val) / (||g||^2 + eps)
        g_norm_sq = np.sum(g ** 2)
        if g_norm_sq < 1e-10:
            break
        step = step_size_factor * (ub_val - lb_val) / (g_norm_sq + 1e-10)
        step = max(step, 0.1)
        step = min(step, 10.0)

        lambda_f = lambda_f + step * g
        lambda_f = np.maximum(lambda_f, 0.0)  # Dual variables must be non-negative

        # Reduce step size over time
        step_size_factor *= 0.995

    # If we have a feasible solution, return it
    if best_assignment is not None:
        return best_assignment

    # Otherwise, use the last assignment and repair
    assignment = np.argmin(assign_cost + np.outer(demands, lambda_f), axis=1)
    assignment = _repair_capacity_violations(assignment, demands, capacities, open_cost, assign_cost)
    return assignment


def _repair_capacity_violations(assignment, demands, capacities, open_cost, assign_cost):
    """Repair capacity violations by moving customers to less-loaded facilities."""
    n_cust = len(demands)
    n_fac = len(capacities)

    assignment = assignment.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]

    # For each facility over capacity, move customers to other facilities
    for f in range(n_fac):
        while load[f] > capacities[f]:
            # Find customers at facility f, move the one with smallest increase in cost
            candidates = [i for i in range(n_cust) if assignment[i] == f]
            if not candidates:
                break

            best_i = None
            best_f2 = None
            best_cost_increase = np.inf

            for i in candidates:
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if load[f2] + demands[i] <= capacities[f2]:
                        cost_increase = assign_cost[i, f2] - assign_cost[i, f]
                        # Also consider opening cost if f2 is new
                        if load[f2] == 0:
                            cost_increase += open_cost[f2]
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_i = i
                            best_f2 = f2

            if best_i is None:
                # Can't repair, break
                break

            load[f] -= demands[best_i]
            load[best_f2] += demands[best_i]
            assignment[best_i] = best_f2

    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: try moving customers to other facilities if it improves cost."""
    n_cust = len(demands)
    n_fac = len(capacities)

    assignment = assignment.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]

    current_cost = _compute_cost(assignment, demands, capacities, open_cost, assign_cost)
    improved = True
    max_rounds = 50

    for rnd in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        if not improved:
            break
        improved = False

        for i in range(n_cust):
            if time.time() > deadline - 5.0:
                break
            f_current = assignment[i]
            best_f = f_current
            best_cost = np.inf

            for f in range(n_fac):
                if f == f_current:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue

                # Compute cost change
                new_load = load.copy()
                new_load[f_current] -= demands[i]
                new_load[f] += demands[i]

                # Check if f_current becomes empty (saves opening cost)
                old_open = open_cost[f_current] if new_load[f_current] > 0 else 0.0
                new_open = open_cost[f] if new_load[f] > 0 else 0.0
                # But f_current was already open, so we only save if it becomes empty
                old_total_open = 0.0
                for ff in range(n_fac):
                    if load[ff] > 0:
                        old_total_open += open_cost[ff]
                new_total_open = 0.0
                for ff in range(n_fac):
                    if new_load[ff] > 0:
                        new_total_open += open_cost[ff]

                cost_change = (assign_cost[i, f] - assign_cost[i, f_current]) + (new_total_open - old_total_open)

                if cost_change < best_cost - 1e-9:
                    best_cost = cost_change
                    best_f = f

            if best_f != f_current:
                load[f_current] -= demands[i]
                load[best_f] += demands[i]
                assignment[i] = best_f
                current_cost += best_cost
                improved = True

    return assignment


def _compute_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)

    total = 0.0
    for i in range(n_cust):
        total += assign_cost[i, assignment[i]]
    for f in range(n_fac):
        if any(assignment[i] == f for i in range(n_cust)):
            total += open_cost[f]
    return total


def _is_valid(assignment, demands, capacities):
    """Check if assignment is valid."""
    if assignment is None:
        return False
    if len(assignment) != len(demands):
        return False
    if np.any(assignment < 0) or np.any(assignment >= len(capacities)):
        return False
    if np.any(np.isnan(assignment)):
        return False
    load = np.zeros(len(capacities), dtype=np.int64)
    for i in range(len(demands)):
        load[assignment[i]] += demands[i]
    if np.any(load > capacities):
        return False
    return True


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        placed = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Should not happen, but assign to least loaded
            f = np.argmin(load)
            result[i] = f
            load[f] += demands[i]
    return result
# EVOLVE-BLOCK-END
