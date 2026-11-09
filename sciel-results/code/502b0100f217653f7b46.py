"""Incentive-compatible finite auction design: 1 bidder, 2 goods, values in {0,1,2,3}."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import itertools
    import time

    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Values in {0,1,2,3} for each of 2 goods -> 4*4 = 16 types
    vals = [0, 1, 2, 3]
    types = list(itertools.product(vals, vals))  # (v1, v2)
    type_index = {t: i for i, t in enumerate(types)}
    n_types = len(types)  # 16
    n_goods = 2

    # Parse distribution
    # distribution_spec could be a dict mapping type->prob, or list of probs, etc.
    if isinstance(distribution_spec, dict):
        mu = np.zeros(n_types)
        for t, p in distribution_spec.items():
            if t in type_index:
                mu[type_index[t]] = p
        mu = mu / mu.sum() if mu.sum() > 0 else np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        mu = np.array(distribution_spec, dtype=float).flatten()
        if len(mu) == n_types:
            mu = mu / mu.sum() if mu.sum() > 0 else np.ones(n_types) / n_types
        else:
            mu = np.ones(n_types) / n_types
    else:
        mu = np.ones(n_types) / n_types

    # Support: list of allowed report types
    if isinstance(support, (list, tuple)):
        if all(isinstance(x, (list, tuple)) for x in support):
            allowed_types = [tuple(x) for x in support]
        else:
            allowed_types = list(support)
    else:
        allowed_types = types

    # Ensure all 16 types are in allowed reports (task says "every allowed report is enumerated")
    # The output must be (16,2) allocations and (16,) payments, so we assume all 16 types are reports
    # Let's just use all 16 types as reports
    reports = types  # all 16 types are reports
    n_reports = len(reports)

    # LP formulation:
    # Variables: a(t, g) for t in reports, g in goods (16*2 = 32 vars), p(t) for t in reports (16 vars)
    # Total: 48 variables
    # Maximize: sum_t mu(t) * p(t)
    # Subject to:
    # 1. 0 <= a(t,g) <= 1 for all t,g
    # 2. p(t) >= 0 for all t
    # 3. IR: v(t).a(t) - p(t) >= 0 for all t  =>  p(t) <= v(t).a(t)
    # 4. IC: for all t, t': v(t).a(t) - p(t) >= v(t).a(t') - p(t')
    #    => p(t) - p(t') <= v(t).a(t) - v(t).a(t')
    #    => p(t) - p(t') - v(t).a(t) + v(t).a(t') <= 0

    # Variable ordering: [a(0,0), a(0,1), a(1,0), a(1,1), ..., a(15,0), a(15,1), p(0), ..., p(15)]
    n_a = n_reports * n_goods  # 32
    n_p = n_reports  # 16
    n_vars = n_a + n_p  # 48

    def a_idx(t, g):
        return t * n_goods + g

    def p_idx(t):
        return n_a + t

    # Objective: maximize sum mu(t)*p(t) -> minimize -sum mu(t)*p(t)
    c = np.zeros(n_vars)
    for t in range(n_reports):
        c[p_idx(t)] = -mu[t]

    # Inequality constraints: A_ub @ x <= b_ub
    # 1. IR: p(t) - v(t).a(t) <= 0
    #    => -v(t,0)*a(t,0) - v(t,1)*a(t,1) + p(t) <= 0
    # 2. IC: p(t) - p(t') - v(t).a(t) + v(t).a(t') <= 0
    #    => -v(t,0)*a(t,0) - v(t,1)*a(t,1) + v(t,0)*a(t',0) + v(t,1)*a(t',1) + p(t) - p(t') <= 0
    # 3. a(t,g) >= 0 and a(t,g) <= 1 (bounds)
    # 4. p(t) >= 0 (bounds)

    A_rows = []
    b_rows = []

    # IR constraints: for each t
    for t in range(n_reports):
        row = np.zeros(n_vars)
        row[a_idx(t, 0)] = -reports[t][0]
        row[a_idx(t, 1)] = -reports[t][1]
        row[p_idx(t)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # IC constraints: for each pair (t, t')
    for t in range(n_reports):
        for tp in range(n_reports):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[a_idx(t, 0)] = -reports[t][0]
            row[a_idx(t, 1)] = -reports[t][1]
            row[a_idx(tp, 0)] = reports[t][0]
            row[a_idx(tp, 1)] = reports[t][1]
            row[p_idx(t)] = 1.0
            row[p_idx(tp)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds
    bounds = []
    for t in range(n_reports):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_reports):
        bounds.append((0.0, None))

    # Solve LP
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x = result.x
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)
        for t in range(n_reports):
            for g in range(n_goods):
                allocations[t, g] = max(0.0, min(1.0, x[a_idx(t, g)]))
            payments[t] = max(0.0, x[p_idx(t)])
    else:
        # Fallback: no-trade mechanism (always allocate 0, pay 0) - satisfies all constraints
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)

    # Verify constraints and fix any numerical issues
    # Check IC and IR, adjust if needed
    def verify_and_fix(allocations, payments, reports, mu, tol=1e-8):
        n = len(reports)
        v = np.array([list(r) for r in reports], dtype=float)
        # IR: v(t).a(t) - p(t) >= 0
        for t in range(n):
            util = v[t] @ allocations[t] - payments[t]
            if util < -tol:
                # Fix: reduce payment
                payments[t] = max(0.0, v[t] @ allocations[t])
        # IC: for all t, t': v(t).a(t) - p(t) >= v(t).a(t') - p(t')
        # This is harder to fix; if violated, we may need to reduce allocations
        # Simple fix: if IC is violated, set allocations to be "monotone" 
        # For now, just check and if violated, fall back to a safe mechanism
        max_violation = 0.0
        for t in range(n):
            for tp in range(n):
                lhs = v[t] @ allocations[t] - payments[t]
                rhs = v[t] @ allocations[tp] - payments[tp]
                if lhs - rhs < -tol:
                    max_violation = max(max_violation, rhs - lhs)
        return max_violation

    violation = verify_and_fix(allocations, payments, reports, mu)
    if violation > 1e-8:
        # Fallback: use a simple truthful mechanism
        # Allocate good g to type t with probability 1 if v(t,g) >= threshold
        # Simplest: no trade
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)

    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
