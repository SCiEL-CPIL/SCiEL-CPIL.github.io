"""Incentive-compatible finite auction design - LP with tight constraints and polish."""
import numpy as np
from scipy.optimize import linprog
import time


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification
        support: Support specification
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85.0

    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    V = np.array(types, dtype=np.float64)

    probs = _parse_distribution(distribution_spec, n_types)

    if time.time() - start_time > DEADLINE:
        return _fallback_result()

    # Solve the LP directly with allocations and payments
    result = _solve_direct_lp(probs, types, n_types, n_goods, start_time, DEADLINE)
    if result is None:
        return _fallback_result()

    x = result['allocations'].copy()
    p = result['payments'].copy()

    # Polish: increase payments where possible
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        p = _polish_payments(x, p, probs, types, n_types, n_goods, start_time, DEADLINE)

    # Final verification
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


def _solve_direct_lp(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve LP with variables x_{t,g} (allocations) and p_t (payments).
    Objective: Maximize sum_t probs[t] * p_t
    Constraints:
      IR: p_t <= v_t . x_t
      IC: p_t - p_{t'} <= v_t . (x_t - x_{t'})
      0 <= x_{t,g} <= 1, p_t >= 0
    """
    n = n_types
    n_vars = n * n_goods + n
    x_start = 0
    p_start = n * n_goods

    c = np.zeros(n_vars)
    for t in range(n):
        c[p_start + t] = -probs[t]

    A_rows = []
    b_rows = []

    # IR: p_t - v_t . x_t <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[p_start + t] = 1.0
        for g in range(n_goods):
            row[x_start + t * n_goods + g] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)

    # IC: p_t - p_{t'} - v_t . (x_t - x_{t'}) <= 0
    # => p_t - p_{t'} - sum_g v_{t,g} x_{t,g} + sum_g v_{t,g} x_{t',g} <= 0
    for t in range(n):
        for t_prime in range(n):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[p_start + t] = 1.0
            row[p_start + t_prime] = -1.0
            for g in range(n_goods):
                row[x_start + t * n_goods + g] = -types[t][g]
                row[x_start + t_prime * n_goods + g] = types[t][g]
            A_rows.append(row)
            b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    bounds = [(0, 1)] * (n * n_goods) + [(0, None)] * n

    time_left = deadline - (time.time() - start_time) - 5.0
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

        z = res.x
        x = z[x_start:x_start + n * n_goods].reshape(n, n_goods)
        p = z[p_start:p_start + n]

        x = np.clip(x, 0, 1)
        p = np.maximum(p, 0.0)

        # Enforce IR strictly
        vdotx = np.array([np.dot(x[t], types[t]) for t in range(n)])
        p = np.minimum(p, vdotx)
        p = np.maximum(p, 0.0)

        return {'allocations': x, 'payments': p}
    except Exception:
        return None


def _polish_payments(x, p, probs, types, n_types, n_goods, start_time, deadline):
    """
    Polish payments by finding the maximum feasible payment for each type,
    iterating until convergence.
    """
    V = np.array(types, dtype=np.float64)
    x = x.copy()
    p = p.copy()

    vdotx = np.array([np.dot(x[t], types[t]) for t in range(n_types)])

    def max_payment_for_type(t, current_p):
        """Find maximum p[t] given other payments fixed."""
        ir_limit = vdotx[t]
        ic_limit = np.inf
        for t_prime in range(n_types):
            if t_prime == t:
                continue
            # IC: p[t] - p[t'] <= v_t . (x_t - x_{t'})
            # => p[t] <= p[t'] + v_t . (x_t - x_{t'})
            rhs = current_p[t_prime] + np.dot(V[t], x[t] - x[t_prime])
            ic_limit = min(ic_limit, rhs)
        return max(0.0, min(ir_limit, ic_limit))

    max_rounds = 50
    for _round in range(max_rounds):
        if time.time() - start_time > deadline - 3.0:
            break

        improved = False
        for t in range(n_types):
            max_p_t = max_payment_for_type(t, p)
            if max_p_t > p[t] + 1e-14:
                p[t] = max_p_t
                improved = True

        if not improved:
            break

    return p


def _verify_and_repair(x, p, probs, types, n_types, n_goods):
    """
    Verify constraints and repair any numerical violations.
    """
    tol = 1e-9
    V = np.array(types, dtype=np.float64)

    x = np.clip(x, 0, 1)

    # Compute minimal payments from x using the payment identity
    # For monotone allocations, the minimal payment is:
    # p[t] = v_t . x_t - U[t] where U[t] is the utility
    # Utility is determined by IC and IR
    
    # Use iterative method to find consistent utilities
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

    # Check if this is feasible
    if _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-8):
        # Use the better of p and p_new
        rev_p = np.dot(probs, p)
        rev_pnew = np.dot(probs, p_new)
        if rev_pnew >= rev_p:
            p = p_new
    else:
        # Try to fix p
        p = np.clip(p, 0, None)
        if not _check_feasibility(x, p, types, n_types, n_goods, tol=1e-8):
            # Fallback: use minimal payments
            p = p_new
            if not _check_feasibility(x, p, types, n_types, n_goods, tol=1e-8):
                x = np.zeros((n_types, n_goods))
                p = np.zeros(n_types)

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


def _fallback_result():
    """Return a valid fallback result (no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
