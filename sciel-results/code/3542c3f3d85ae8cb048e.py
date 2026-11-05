import time
import numpy as np
from scipy.optimize import linprog, minimize

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2

    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)

    # 1. Solve allocation+utility LP
    result = _solve_allocation_with_utility_lp(probs, types, n_types, n_goods, deadline)

    if time.time() > deadline - 8.0:
        return _simple_safe_mechanism(types)

    if result is None:
        return _simple_safe_mechanism(types)

    final_alloc, final_pay = result

    # 2. Clip to valid ranges
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    final_pay = np.maximum(final_pay, 0.0)

    # 3. Multi-start SLSQP polish for high-precision refinement
    if time.time() < deadline - 4.0:
        final_alloc, final_pay = _polish_multistart_slsqp(final_alloc, final_pay, probs, types, deadline)

    # Final safety: verify and repair if needed
    final_alloc, final_pay = _verify_and_repair(final_alloc, final_pay, probs, types)

    return {
        'allocations': final_alloc,
        'payments': final_pay
    }


def _parse_distribution(distribution_spec, support, types, n_types):
    """Robustly parse distribution spec to get type probabilities."""
    if distribution_spec is None:
        return np.ones(n_types) / n_types

    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
            return np.ones(n_types) / n_types

    if isinstance(distribution_spec, dict):
        for key in ['probs', 'distribution', 'type_probs', 'probabilities', 'p']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        arr = np.maximum(arr, 0)
                        if arr.sum() > 0:
                            return arr / arr.sum()
                elif isinstance(val, dict):
                    probs = np.zeros(n_types)
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()

        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
                found_any = True
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()

    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    return np.ones(n_types) / n_types


def _solve_allocation_with_utility_lp(probs, types, n_types, n_goods, deadline):
    """
    Solve the mechanism design LP with variables x[i][g] and U[i].
    Objective: minimize -sum_i probs[i] * (v_i . x_i - U[i])
    """
    def var_x(i, g):
        return i * n_goods + g

    def var_U(i):
        return n_types * n_goods + i

    n_vars = n_types * n_goods + n_types

    c = np.zeros(n_vars)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for g in range(n_goods):
            c[var_x(i, g)] = -probs[i] * vi[g]
        c[var_U(i)] = probs[i]

    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir

    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    idx = 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            A_ub[idx, var_U(i)] = -1.0
            A_ub[idx, var_U(j)] = 1.0
            for g in range(n_goods):
                A_ub[idx, var_x(j, g)] = vi[g] - vj[g]
            idx += 1

    for i in range(n_types):
        A_ub[idx, var_U(i)] = -1.0
        idx += 1

    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types

    remaining = max(5.0, deadline - time.time() - 5.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": remaining})

        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]

            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - utilities[i]
                payments[i] = max(0.0, p_i)

            return allocations, payments
        else:
            return None
    except Exception:
        return None


def _compute_revenue(allocations, utilities, probs, types):
    """Compute expected revenue given allocations and utilities."""
    n_types = len(types)
    rev = 0.0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = allocations[i]
        rev += probs[i] * (np.dot(vi, xi) - utilities[i])
    return rev


def _check_feasibility(allocations, utilities, probs, types, tol=1e-7):
    """Check IC and IR feasibility."""
    n_types = len(types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = allocations[i]
        ui = utilities[i]

        # IR
        if ui < -tol:
            return False

        # IC: u_i(i) >= u_i(j) for all j
        for j in range(n_types):
            vj = np.array(types[j], dtype=float)
            xj = allocations[j]
            uj_as_seen_by_i = np.dot(vi, xj) - (np.dot(vj, xj) - utilities[j])
            if ui - uj_as_seen_by_i < -tol:
                return False
    return True


def _polish_multistart_slsqp(allocations, payments, probs, types, deadline):
    """
    Multi-start SLSQP polish: try multiple initial feasible points and keep the best.
    """
    n_types = len(types)
    n_goods = 2
    n_alloc = n_types * n_goods
    n_util = n_types
    n_total = n_alloc + n_util

    # Build constraint functions once
    def make_constraints():
        constraints = []
        # IC: U_i - U_j >= vi.xj - vj.xj  =>  -(U_i - U_j - vi.xj + vj.xj) <= 0
        #     i.e., U_j - U_i + (vj - vi).xj <= 0  =>  ineq: -(U_j - U_i + (vj-vi).xj) >= 0
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            for j in range(n_types):
                if i == j:
                    continue
                vj = np.array(types[j], dtype=float)
                diff = vj - vi
                def ic_fun(z, i=i, j=j, diff=diff):
                    util = z[n_alloc:]
                    alloc = z[:n_alloc].reshape(n_types, n_goods)
                    xj = alloc[j]
                    return -(util[j] - util[i] + np.dot(diff, xj))
                constraints.append({'type': 'ineq', 'fun': ic_fun})

        # IR: U_i >= 0
        for i in range(n_types):
            def ir_fun(z, i=i):
                return z[n_alloc + i]
            constraints.append({'type': 'ineq', 'fun': ir_fun})

        return constraints

    constraints = make_constraints()
    bounds = [(0, 1)] * n_alloc + [(0, None)] * n_util

    def objective(z):
        alloc = z[:n_alloc].reshape(n_types, n_goods)
        util = z[n_alloc:]
        rev = 0.0
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            rev += probs[i] * (np.dot(vi, alloc[i]) - util[i])
        return -rev

    # Generate multiple starting points
    start_points = []

    # Start 1: LP solution
    utils0 = np.zeros(n_types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        utils0[i] = np.dot(vi, allocations[i]) - payments[i]
    z0 = np.concatenate([allocations.flatten(), utils0])
    start_points.append(z0)

    # Start 2: Greedy allocation (allocate good g to types with highest value for g)
    greedy_alloc = np.zeros((n_types, n_goods))
    for g in range(n_goods):
        values = np.array([types[i][g] for i in range(n_types)], dtype=float)
        # Allocate to types with value >= 2 (top half)
        greedy_alloc[:, g] = (values >= 2).astype(float)
    # Compute utilities for greedy
    greedy_utils = np.zeros(n_types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        greedy_utils[i] = np.dot(vi, greedy_alloc[i])
    # Ensure IR: U_i >= 0 already satisfied
    # Check IC and adjust if needed
    z_greedy = np.concatenate([greedy_alloc.flatten(), greedy_utils])
    start_points.append(z_greedy)

    # Start 3: Slightly perturbed LP solution
    if len(start_points) > 0:
        z_pert = z0.copy()
        z_pert[:n_alloc] += np.random.uniform(-0.01, 0.01, size=n_alloc)
        z_pert[:n_alloc] = np.clip(z_pert[:n_alloc], 0.0, 1.0)
        start_points.append(z_pert)

    # Start 4: Uniform allocation
    uniform_alloc = np.full((n_types, n_goods), 0.5)
    uniform_utils = np.zeros(n_types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        uniform_utils[i] = np.dot(vi, uniform_alloc[i])
    z_uniform = np.concatenate([uniform_alloc.flatten(), uniform_utils])
    start_points.append(z_uniform)

    # Deduplicate and filter feasible starts
    feasible_starts = []
    seen = set()
    for z in start_points:
        # Round to avoid duplicates
        key = tuple(np.round(z, 6))
        if key in seen:
            continue
        seen.add(key)

        alloc = z[:n_alloc].reshape(n_types, n_goods)
        util = z[n_alloc:]
        if _check_feasibility(alloc, util, probs, types, tol=1e-5):
            feasible_starts.append(z)

    # If no feasible starts, use the LP solution as-is
    if not feasible_starts:
        feasible_starts = [z0]

    # Run SLSQP from each feasible start, keep best
    best_z = None
    best_rev = -np.inf

    remaining_per_start = max(1.0, (deadline - time.time() - 2.0) / len(feasible_starts))
    max_iter = max(50, int(remaining_per_start * 20))

    for idx, z_start in enumerate(feasible_starts):
        if time.time() > deadline - 2.0:
            break

        try:
            res = minimize(objective, z_start, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})

            if res.success:
                alloc = res.x[:n_alloc].reshape(n_types, n_goods)
                util = res.x[n_alloc:]
                alloc = np.clip(alloc, 0.0, 1.0)
                util = np.maximum(util, 0.0)

                if _check_feasibility(alloc, util, probs, types, tol=1e-8):
                    rev = _compute_revenue(alloc, util, probs, types)
                    if rev > best_rev:
                        best_rev = rev
                        best_z = res.x
        except Exception:
            continue

    if best_z is not None:
        alloc = best_z[:n_alloc].reshape(n_types, n_goods)
        util = best_z[n_alloc:]
        alloc = np.clip(alloc, 0.0, 1.0)
        util = np.maximum(util, 0.0)

        pay = np.zeros(n_types)
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            pay[i] = max(0.0, np.dot(vi, alloc[i]) - util[i])

        return alloc, pay
    else:
        return allocations, payments


def _verify_and_repair(allocations, payments, probs, types):
    """
    Verify IC/IR and non-negativity. Repair by recomputing payments via LP if violated.
    """
    n_types = len(types)
    tol = 1e-8

    violations = False
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = np.array(allocations[i], dtype=float)
        ui = np.dot(vi, xi) - payments[i]

        if ui < -tol:
            violations = True
            break

        for j in range(n_types):
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            uj = np.dot(vi, xj) - payments[j]
            if ui - uj < -tol:
                violations = True
                break
        if violations:
            break

    if not violations:
        return allocations, payments

    # Repair: recompute payments via utility LP
    result = _compute_payments_lp(allocations, probs, types)
    if result is not None:
        return result

    return np.zeros((n_types, 2)), np.zeros(n_types)


def _compute_payments_lp(allocations, probs, types):
    """Compute optimal payments (minimize expected utility) given fixed allocation."""
    n_types = len(types)
    c = probs.copy()
    n_vars = n_types
    n_ineq = n_types + n_types * (n_types - 1)

    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    idx = 0
    for i in range(n_types):
        A_ub[idx, i] = -1.0
        idx += 1

    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1

    bounds = [(0, None)] * n_types

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 10})

        if res.success:
            U = np.maximum(res.x, 0.0)
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                payments[i] = max(0.0, np.dot(vi, xi) - U[i])
            return allocations, payments
        return None
    except Exception:
        return None


def _simple_safe_mechanism(types):
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
