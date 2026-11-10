"""Incentive-compatible finite auction design - two-phase LP with payment optimization."""
import numpy as np
from scipy.optimize import linprog
import time


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification (list of 16 probabilities or similar)
        support: Support specification (16 types, 2 goods)
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    V = np.array(types, dtype=np.float64)

    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)

    if time.time() - start_time > DEADLINE:
        return _fallback_result()

    # Phase 1: Solve joint LP for allocations and utilities
    result = _solve_joint_lp(probs, types, n_types, n_goods, start_time, DEADLINE)
    if result is None:
        return _fallback_result()

    x = result['allocations'].copy()
    p = result['payments'].copy()

    # Phase 2: Enforce monotonicity on allocations
    x = _enforce_monotonicity(x, types, n_types, n_goods)

    # Phase 3: Fix allocations, solve payment-only LP to maximize payments
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10.0:
        p_new = _solve_payment_lp(x, probs, types, n_types, n_goods, start_time, DEADLINE)
        if p_new is not None:
            p = p_new

    # Phase 4: Payment probe refinement (squeeze residual revenue)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        x, p = _payment_probe_refinement(x, p, probs, types, n_types, n_goods, start_time, DEADLINE)

    # Phase 5: Final verification and repair
    x, p = _verify_and_repair(x, p, probs, types, n_types, n_goods)

    return {
        'allocations': x,
        'payments': p
    }


def _parse_distribution(distribution_spec, n_types):
    """Parse distribution specification into probability vector."""
    probs = np.zeros(n_types)

    if distribution_spec is None:
        return np.ones(n_types) / n_types

    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            d = distribution_spec['distribution']
            if isinstance(d, str) and d == 'uniform':
                return np.ones(n_types) / n_types
            elif isinstance(d, str) and d == 'geometric':
                types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
                base = 0.7
                probs = np.array([base ** types[i][0] * base ** types[i][1] for i in range(n_types)])
            else:
                probs = np.array(d, dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (4, 4):
            probs = arr.flatten()
        elif arr.size == n_types:
            probs = arr.flatten()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    probs = np.asarray(probs, dtype=np.float64).flatten()
    if probs.shape[0] != n_types:
        probs = np.ones(n_types) / n_types

    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 1e-15:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types

    return probs


def _enforce_monotonicity(x, types, n_types, n_goods):
    """
    Enforce monotonicity of allocations: for additive values, IC requires
    that allocation of good g is non-decreasing in value of good g.
    Average adjacent types to enforce this.
    """
    x = x.copy()
    for _ in range(10):  # Iterate to convergence
        changed = False
        # Monotonicity in v1 for good 0
        for v1 in range(3):
            for v2 in range(4):
                t_curr = v1 * 4 + v2
                t_next = (v1 + 1) * 4 + v2
                if x[t_curr, 0] > x[t_next, 0] + 1e-12:
                    avg = (x[t_curr, 0] + x[t_next, 0]) / 2
                    x[t_curr, 0] = avg
                    x[t_next, 0] = avg
                    changed = True
        # Monotonicity in v2 for good 1
        for v1 in range(4):
            for v2 in range(3):
                t_curr = v1 * 4 + v2
                t_next = v1 * 4 + (v2 + 1)
                if x[t_curr, 1] > x[t_next, 1] + 1e-12:
                    avg = (x[t_curr, 1] + x[t_next, 1]) / 2
                    x[t_curr, 1] = avg
                    x[t_next, 1] = avg
                    changed = True
        if not changed:
            break
    return np.clip(x, 0, 1)


def _solve_joint_lp(probs, types, n_types, n_goods, start_time, deadline):
    """
    Phase 1: Solve the joint LP with allocations and utilities.
    Variables: x_{t,g} (allocations) and u_t (utilities).
    Objective: Maximize expected revenue = sum_t probs[t] * (dot(v_t, x_t) - u_t)
    """
    n_vars = n_types * n_goods + n_types
    x_start = 0
    u_start = n_types * n_goods

    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[x_start + t * n_goods + g] = -probs[t] * types[t][g]
        c[u_start + t] = probs[t]

    A_rows = []
    b_rows = []

    # IR: -u_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # IC: u_t >= u_{t'} + dot(v_t - v_{t'}, x_{t'})
    # => -u_t + u_{t'} + sum_g (v_{t,g} - v_{t',g}) x_{t',g} <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[u_start + t] = -1.0
            row[u_start + t_prime] = 1.0
            for g in range(n_goods):
                coeff = types[t][g] - types[t_prime][g]
                if coeff != 0:
                    row[x_start + t_prime * n_goods + g] = coeff
            A_rows.append(row)
            b_rows.append(0.0)

    # Non-negative payments: u_t - sum_g v_{t,g} x_{t,g} <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = 1.0
        for g in range(n_goods):
            row[x_start + t * n_goods + g] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types

    time_left = deadline - (time.time() - start_time) - 5.0
    if time_left < 1.0:
        return None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'time_limit': max(1.0, time_left)})
        if not res.success or res.x is None:
            return None

        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        u = z[u_start:u_start + n_types]

        x = np.clip(x, 0, 1)
        p = np.zeros(n_types)
        for t in range(n_types):
            p[t] = np.dot(x[t], types[t]) - u[t]
        p = np.maximum(p, 0.0)

        return {'allocations': x, 'payments': p}
    except Exception:
        return None


def _solve_payment_lp(x, probs, types, n_types, n_goods, start_time, deadline):
    """
    Phase 3: Given fixed allocations x, solve for maximum payments.
    Variables: p_t (payments).
    Objective: Maximize sum_t probs[t] * p_t
    Subject to:
      IC: p[t] - p[t'] <= v_t . (x_t - x_{t'})  for all t != t'
      IR: p[t] <= v_t . x_t
      p[t] >= 0
    """
    n = n_types
    c = np.zeros(n)
    for t in range(n):
        c[t] = -probs[t]  # minimize negative revenue

    A_ub = []
    b_ub = []

    # IC constraints: p[t] - p[t'] - v_t . (x_t - x_{t'}) <= 0
    for t in range(n):
        for t_prime in range(n):
            if t == t_prime:
                continue
            row = np.zeros(n)
            row[t] = 1.0
            row[t_prime] = -1.0
            rhs = np.dot(types[t], x[t] - x[t_prime])
            A_ub.append(row)
            b_ub.append(rhs)

    # IR constraints: p[t] - v_t . x_t <= 0
    for t in range(n):
        row = np.zeros(n)
        row[t] = 1.0
        A_ub.append(row)
        b_ub.append(np.dot(types[t], x[t]))

    # Non-negative payments: -p[t] <= 0
    for t in range(n):
        row = np.zeros(n)
        row[t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n

    time_left = deadline - (time.time() - start_time) - 8.0
    if time_left < 1.0:
        return None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14,
                               'time_limit': max(1.0, time_left)})
        if not res.success or res.x is None:
            return None

        p = np.clip(res.x, 0, None)
        return p
    except Exception:
        return None


def _payment_probe_refinement(x, p, probs, types, n_types, n_goods, start_time, deadline):
    """
    Directly probe for payment increases that maintain IC/IR feasibility.
    For each type t, the max increase is limited by IR and IC constraints.
    """
    V = np.array(types, dtype=np.float64)
    x = x.copy()
    p = p.copy()

    vdotx = np.array([np.dot(x[t], types[t]) for t in range(n_types)])

    def max_payment_increase(t):
        ir_limit = vdotx[t] - p[t]
        ic_limit = np.inf
        for t_prime in range(n_types):
            if t_prime == t:
                continue
            rhs = p[t_prime] + np.dot(V[t], x[t] - x[t_prime])
            ic_limit = min(ic_limit, rhs - p[t])
        return max(0.0, min(ir_limit, ic_limit))

    max_iter = 200
    for iteration in range(max_iter):
        if time.time() - start_time > deadline - 3.0:
            break

        best_t = -1
        best_inc = 0.0
        for t in range(n_types):
            inc = max_payment_increase(t)
            if inc > best_inc:
                best_inc = inc
                best_t = t

        if best_t < 0 or best_inc < 1e-12:
            break

        if iteration < 50:
            frac = 0.5
        elif iteration < 100:
            frac = 0.25
        elif iteration < 150:
            frac = 0.1
        else:
            frac = 0.05

        delta = best_inc * frac
        if delta < 1e-14:
            break

        p[best_t] += delta

        if not _check_feasibility(x, p, types, n_types, n_goods, tol=1e-10):
            p[best_t] -= delta
            break

    return x, p


def _check_feasibility(x, p, types, n_types, n_goods, tol=1e-10):
    """Check if (x, p) satisfies IR, IC, and non-negative payments."""
    V = np.array(types, dtype=np.float64)

    if np.any(p < -tol):
        return False

    for t in range(n_types):
        if p[t] > np.dot(V[t], x[t]) + tol:
            return False

    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            lhs = p[t] - p[t_prime]
            rhs = np.dot(V[t], x[t] - x[t_prime])
            if lhs > rhs + tol:
                return False

    return True


def _verify_and_repair(x, p, probs, types, n_types, n_goods):
    """
    Verify constraints and repair any numerical violations.
    """
    tol = 1e-9
    V = np.array(types, dtype=np.float64)

    x = np.clip(x, 0, 1)

    # Compute minimal utilities from x using Bellman-Ford style
    U = np.zeros(n_types)
    for _ in range(n_types + 1):
        updated = False
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                w = np.dot(V[t] - V[t_prime], x[t_prime])
                new_val = U[t_prime] + w
                if new_val > U[t] + tol:
                    U[t] = new_val
                    updated = True
        for t in range(n_types):
            if U[t] < 0:
                U[t] = 0.0
                updated = True
        if not updated:
            break

    p_new = np.zeros(n_types)
    for t in range(n_types):
        p_new[t] = np.dot(V[t], x[t]) - U[t]

    if np.any(p_new < -tol):
        for t in range(n_types):
            if p_new[t] < -tol:
                vdotx = np.dot(V[t], x[t])
                if vdotx > 1e-12:
                    scale = max(0.0, 1.0 + p_new[t] / vdotx)
                    x[t] *= scale
                    x[t] = np.clip(x[t], 0, 1)
                else:
                    x[t] = np.zeros(n_goods)

        U = np.zeros(n_types)
        for _ in range(n_types + 1):
            updated = False
            for t in range(n_types):
                for t_prime in range(n_types):
                    if t == t_prime:
                        continue
                    w = np.dot(V[t] - V[t_prime], x[t_prime])
                    new_val = U[t_prime] + w
                    if new_val > U[t] + tol:
                        U[t] = new_val
                        updated = True
            for t in range(n_types):
                if U[t] < 0:
                    U[t] = 0.0
                    updated = True
            if not updated:
                break

        p_new = np.zeros(n_types)
        for t in range(n_types):
            p_new[t] = np.dot(V[t], x[t]) - U[t]

    p_new = np.clip(p_new, 0, None)

    # Check regret
    max_regret = 0.0
    for t in range(n_types):
        util_t = np.dot(V[t], x[t]) - p_new[t]
        for tp in range(n_types):
            if tp == t:
                continue
            util_tp = np.dot(V[t], x[tp]) - p_new[tp]
            regret = util_tp - util_t
            if regret > max_regret:
                max_regret = regret

    if max_regret > 1e-8:
        # Try to fix by adjusting payments
        for iteration in range(100):
            changed = False
            for t in range(n_types):
                min_p = 0.0
                for tp in range(n_types):
                    if tp == t:
                        continue
                    required = p_new[tp] + np.dot(V[t], x[t] - x[tp])
                    if required > min_p:
                        min_p = required
                max_p = np.dot(V[t], x[t])
                if min_p > max_p:
                    # Infeasible; reduce allocation
                    scale = max(0.0, max_p / max(min_p, 1e-12)) * 0.99
                    x[t] *= scale
                    x[t] = np.clip(x[t], 0, 1)
                    min_p = np.dot(V[t], x[t])
                if min_p > p_new[t]:
                    p_new[t] = min_p
                    changed = True
            if not changed:
                break

        # Recompute utilities
        U = np.zeros(n_types)
        for _ in range(n_types + 1):
            updated = False
            for t in range(n_types):
                for t_prime in range(n_types):
                    if t == t_prime:
                        continue
                    w = np.dot(V[t] - V[t_prime], x[t_prime])
                    new_val = U[t_prime] + w
                    if new_val > U[t] + tol:
                        U[t] = new_val
                        updated = True
            for t in range(n_types):
                if U[t] < 0:
                    U[t] = 0.0
                    updated = True
            if not updated:
                break

        p_new = np.zeros(n_types)
        for t in range(n_types):
            p_new[t] = np.dot(V[t], x[t]) - U[t]
        p_new = np.clip(p_new, 0, None)

    # Final feasibility check
    if not _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-8):
        p_new = np.clip(p, 0, None)
        if not _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-8):
            x = np.zeros((n_types, n_goods))
            p_new = np.zeros(n_types)

    return x, p_new


def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
