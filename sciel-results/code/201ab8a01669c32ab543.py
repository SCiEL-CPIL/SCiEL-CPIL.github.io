"""Incentive-compatible finite auction design - optimal LP-based mechanism."""
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
    DEADLINE = 82.0  # Safety margin before 90s limit

    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    V = np.array(types, dtype=np.float64)

    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)

    if time.time() - start_time > DEADLINE:
        return _fallback_result()

    # Phase 1: Solve the LP
    result = _solve_lp_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    if result is None:
        return _fallback_result()

    x = result['allocations'].copy()
    p = result['payments'].copy()

    # Phase 2: High-precision polish LP (tighter tolerances)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15.0:
        polish = _polish_lp(probs, types, n_types, n_goods, x, p, start_time, DEADLINE)
        if polish is not None:
            x, p = polish

    # Phase 3: Direct payment-increase probing (squeeze residual revenue)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        x, p = _payment_probe_refinement(x, p, probs, types, n_types, n_goods, start_time, DEADLINE)

    # Phase 4: Final verification and repair
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


def _solve_lp_auction(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the auction problem using a Linear Programming formulation.
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

    time_left = deadline - (time.time() - start_time) - 3.0
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

        # Quick verification
        x, p = _verify_and_repair(x, p, probs, types, n_types, n_goods)
        return {'allocations': x, 'payments': p}
    except Exception:
        return None


def _polish_lp(probs, types, n_types, n_goods, x_init, p_init, start_time, deadline):
    """
    Re-solve the LP with tighter tolerances, using the initial solution as a
    warm start (HiGHS doesn't support warm start directly, but we can set
    tighter tolerances and check if the solution improves).
    """
    n_vars = n_types * n_goods + n_types
    x_start = 0
    u_start = n_types * n_goods

    # Compute initial utilities from x_init
    u_init = np.zeros(n_types)
    for t in range(n_types):
        u_init[t] = np.dot(x_init[t], types[t]) - p_init[t]
    u_init = np.maximum(u_init, 0.0)

    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[x_start + t * n_goods + g] = -probs[t] * types[t][g]
        c[u_start + t] = probs[t]

    A_rows = []
    b_rows = []

    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

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
                               'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14,
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

        # Compare revenue: only accept if better
        rev_new = np.sum(probs * (x @ np.array(types, dtype=np.float64).T - p))
        rev_old = np.sum(probs * (x_init @ np.array(types, dtype=np.float64).T - p_init))
        if rev_new > rev_old - 1e-10:
            x, p = _verify_and_repair(x, p, probs, types, n_types, n_goods)
            return (x, p)
    except Exception:
        pass

    return None


def _payment_probe_refinement(x, p, probs, types, n_types, n_goods, start_time, deadline):
    """
    Directly probe for payment increases that maintain IC/IR feasibility.
    The key insight: if we can increase payment for type t by delta while
    keeping all IC/IR constraints satisfied, we gain revenue.
    
    For each type t, the maximum we can increase p[t] is limited by:
    - IR: p[t] <= v_t . x_t  => max increase = v_t . x_t - p[t]
    - IC (as reporter t): p[t] - p[t'] <= v_t . (x_t - x_{t'}) for all t'
      => p[t] <= p[t'] + v_t . (x_t - x_{t'}) for all t'
      => max p[t] = min_{t'} (p[t'] + v_t . (x_t - x_{t'}))
    - IC (as t' reporting as t): p[t'] - p[t] <= v_{t'} . (x_{t'} - x_t) for all t'
      => p[t] >= p[t'] - v_{t'} . (x_{t'} - x_t) for all t'
      (This is a LOWER bound on p[t], so increasing p[t] is fine as long as
       we don't violate the upper bounds from other types' IC constraints.)
    
    Actually, increasing p[t] only affects:
    - IC where t is the true type: p[t] - p[t'] <= v_t . (x_t - x_{t'})
      Increasing p[t] makes LHS larger, so this is the binding constraint.
    - IC where t is the reported type: p[t'] - p[t] <= v_{t'} . (x_{t'} - x_t)
      Increasing p[t] makes LHS smaller, so this is relaxed.
    
    So the max increase for p[t] is:
      delta_max[t] = min(
          v_t . x_t - p[t],                              # IR
          min_{t' != t} (p[t'] + v_t . (x_t - x_{t'}) - p[t])  # IC as true type
      )
    
    We greedily increase payments in order of highest marginal revenue gain
    (which is just probs[t] * delta since revenue = sum probs[t] * (v_t.x_t - p_t),
     wait no: revenue = sum probs[t] * (v_t.x_t - p_t), so increasing p_t DECREASES revenue!
    
    Wait, I need to re-read the problem. Revenue = sum_t pi_t * p_t.
    So we want to INCREASE payments to increase revenue!
    
    Let me re-check: "Maximize expected revenue" and revenue = sum pi_t * payment_t.
    Yes! So we want to push payments UP as much as IC/IR allows.
    """
    V = np.array(types, dtype=np.float64)
    x = x.copy()
    p = p.copy()

    # Precompute v_t . x_t for all t
    vdotx = np.array([np.dot(x[t], types[t]) for t in range(n_types)])

    def max_payment_increase(t):
        """Compute how much we can increase p[t]."""
        # IR constraint: p[t] <= v_t . x_t
        ir_limit = vdotx[t] - p[t]

        # IC constraints where t is the true type:
        # p[t] - p[t'] <= v_t . (x_t - x_{t'})
        # => p[t] <= p[t'] + v_t . (x_t - x_{t'})
        ic_limit = np.inf
        for t_prime in range(n_types):
            if t_prime == t:
                continue
            rhs = p[t_prime] + np.dot(V[t], x[t] - x[t_prime])
            ic_limit = min(ic_limit, rhs - p[t])

        return max(0.0, min(ir_limit, ic_limit))

    # Greedy: repeatedly find the type with the most room to increase,
    # increase it by a small step, and repeat.
    # Since increasing p[t] relaxes IC constraints where t is the reported type,
    # but tightens where t is the true type, we need to be careful.
    # However, the IC constraint p[t'] - p[t] <= v_{t'}.(x_{t'}-x_t) is relaxed
    # when p[t] increases, so it's safe.
    # The only constraints that get tighter are those where t is the true type.
    
    # Strategy: for each type, compute the max increase, then increase by a
    # fraction of it. Repeat a few times.
    
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() - start_time > deadline - 3.0:
            break

        # Recompute vdotx (x hasn't changed, so this is cached)
        # Find the type with the largest possible increase
        best_t = -1
        best_inc = 0.0
        for t in range(n_types):
            inc = max_payment_increase(t)
            if inc > best_inc:
                best_inc = inc
                best_t = t

        if best_t < 0 or best_inc < 1e-12:
            break

        # Increase by a fraction (start aggressive, then finer)
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

        # Verify feasibility after update
        if not _check_feasibility(x, p, types, n_types, n_goods, tol=1e-10):
            # Rollback
            p[best_t] -= delta
            break

    return x, p


def _check_feasibility(x, p, types, n_types, n_goods, tol=1e-10):
    """Check if (x, p) satisfies IR, IC, and non-negative payments."""
    V = np.array(types, dtype=np.float64)

    # Non-negative payments
    if np.any(p < -tol):
        return False

    # IR: p[t] <= v_t . x_t
    for t in range(n_types):
        if p[t] > np.dot(V[t], x[t]) + tol:
            return False

    # IC: p[t] - p[t'] <= v_t . (x_t - x_{t'})
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
    Returns repaired (x, p).
    """
    tol = 1e-9
    V = np.array(types, dtype=np.float64)

    # Clip allocations
    x = np.clip(x, 0, 1)

    # Recompute minimal utilities from x using IC/IR (longest path / Bellman-Ford style)
    # u_t = max(0, max_{t'} (u_{t'} + v_t . (x_t - x_{t'})))
    # This is the minimum utility that satisfies IC and IR.
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
        # IR: U[t] >= 0
        for t in range(n_types):
            if U[t] < 0:
                U[t] = 0.0
                updated = True
        if not updated:
            break

    # Recompute payments: p_t = v_t . x_t - u_t
    p_new = np.zeros(n_types)
    for t in range(n_types):
        p_new[t] = np.dot(V[t], x[t]) - U[t]

    # If payments are negative, we need to fix
    if np.any(p_new < -tol):
        # Scale down allocations for types with negative payments
        for t in range(n_types):
            if p_new[t] < -tol:
                vdotx = np.dot(V[t], x[t])
                if vdotx > 1e-12:
                    scale = max(0.0, 1.0 + p_new[t] / vdotx)
                    x[t] *= scale
                    x[t] = np.clip(x[t], 0, 1)
                else:
                    x[t] = np.zeros(n_goods)

        # Recompute
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

    # Final clip
    p_new = np.clip(p_new, 0, None)

    # Final feasibility check
    if not _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-8):
        # If still infeasible, use the original p clipped
        p_new = np.clip(p, 0, None)
        if not _check_feasibility(x, p_new, types, n_types, n_goods, tol=1e-8):
            # Last resort: zero mechanism
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
