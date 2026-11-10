import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
    else:
        probs = probs / total

    # LP variables:
    # x[0:16]   = q1[t]  (allocation prob for good 1 to type t)
    # x[16:32]  = q2[t]  (allocation prob for good 2 to type t)
    # x[32:48]  = p[t]   (payment for type t)

    # Objective: maximize sum_t probs[t] * p[t]
    # => minimize -sum_t probs[t] * p[t]
    c = np.zeros(48)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_ub = []
    b_ub = []

    # IC constraints: for each ordered pair (t, t'), t != t'
    # p[t] - p[t'] - v1[t]*q1[t] + v1[t]*q1[t'] - v2[t]*q2[t] + v2[t]*q2[t'] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        for tp in range(n):
            if tp == t:
                continue
            row = np.zeros(48)
            row[t] = -v1t
            row[tp] = v1t
            row[16 + t] = -v2t
            row[16 + tp] = v2t
            row[32 + t] = 1.0
            row[32 + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: p[t] - v1[t]*q1[t] - v2[t]*q2[t] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        row = np.zeros(48)
        row[t] = -v1t
        row[16 + t] = -v2t
        row[32 + t] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Nonnegative payments: -p[t] <= 0
    for t in range(n):
        row = np.zeros(48)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Bounds
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Check time before solving
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    try:
        options = {'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                   'primal_feasibility_tolerance': 1e-12, 'time_limit': 30.0}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 2:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if result.success and result.x is not None:
        q1 = np.clip(result.x[0:16], 0, 1)
        q2 = np.clip(result.x[16:32], 0, 1)
        p = np.clip(result.x[32:48], 0, None)

        # Final polish: re-solve with tighter tolerances using current solution as warm start
        if time.time() < deadline - 10:
            try:
                options_tight = {'presolve': False,
                                  'dual_feasibility_tolerance': 1e-14,
                                  'primal_feasibility_tolerance': 1e-14,
                                  'time_limit': 10.0}
                result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                  method='highs', options=options_tight,
                                  x0=result.x)
                if result2.success and result2.x is not None:
                    q1_new = np.clip(result2.x[0:16], 0, 1)
                    q2_new = np.clip(result2.x[16:32], 0, 1)
                    p_new = np.clip(result2.x[32:48], 0, None)
                    # Use the solution with higher revenue if both are valid
                    rev_orig = np.dot(probs, p)
                    rev_new = np.dot(probs, p_new)
                    if rev_new >= rev_orig - 1e-10:
                        q1, q2, p = q1_new, q2_new, p_new
            except Exception:
                pass

        # Verify constraints and compute regret
        max_regret = 0.0
        for t in range(n):
            v1t, v2t = types[t]
            util_t = v1t * q1[t] + v2t * q2[t] - p[t]
            for tp in range(n):
                if tp == t:
                    continue
                v1tp, v2tp = types[tp]
                util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                regret = util_tp - util_t
                if regret > max_regret:
                    max_regret = regret

        if max_regret > 1e-8:
            # Iterative regret fix: increase payments to ensure IC
            # For each type t, the minimum payment that satisfies IC against all t'
            # is: p[t] >= max over t' of [p[t'] + v1t*q1[t'] + v2t*q2[t'] - v1t*q1[t] - v2t*q2[t]]
            # Also p[t] >= 0 (nonnegativity) and p[t] <= v1t*q1[t] + v2t*q2[t] (IR)
            # We need p[t] >= max(0, max_{t'!=t} [p[t'] + v1t*(q1[t']-q1[t]) + v2t*(q2[t']-q2[t])])
            # But this creates circular dependencies. Use iterative relaxation.
            
            for iteration in range(100):
                if time.time() > deadline - 3:
                    break
                changed = False
                for t in range(n):
                    v1t, v2t = types[t]
                    min_p = 0.0
                    for tp in range(n):
                        if tp == t:
                            continue
                        v1tp, v2tp = types[tp]
                        # IC: p[t] - p[tp] >= v1t*q1[t] + v2t*q2[t] - v1t*q1[tp] - v2t*q2[tp]
                        required = p[tp] + v1t * (q1[t] - q1[tp]) + v2t * (q2[t] - q2[tp])
                        if required > min_p:
                            min_p = required
                    # IR: p[t] <= v1t*q1[t] + v2t*q2[t]
                    max_p = v1t * q1[t] + v2t * q2[t]
                    if min_p > max_p:
                        # Infeasible for this allocation; reduce allocations slightly
                        # For now, set to max_p (may violate IC but is best effort)
                        min_p = max_p
                    if min_p > p[t] + 1e-12:
                        p[t] = min_p
                        changed = True
                if not changed:
                    break

            # Re-clip
            p = np.clip(p, 0, None)

        allocations = np.column_stack([q1, q2])
        payments = p
    else:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)

    return {'allocations': allocations, 'payments': payments}
