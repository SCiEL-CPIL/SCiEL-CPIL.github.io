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
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 2:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if result.success and result.x is not None:
        q1 = np.clip(result.x[0:16], 0, 1)
        q2 = np.clip(result.x[16:32], 0, 1)
        p = np.clip(result.x[32:48], 0, None)

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
            # Regret violation; try to fix by adjusting payments
            # Simple fix: increase payments for types with positive regret
            for t in range(n):
                v1t, v2t = types[t]
                for tp in range(n):
                    if tp == t:
                        continue
                    v1tp, v2tp = types[tp]
                    util_t = v1t * q1[t] + v2t * q2[t] - p[t]
                    util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                    if util_tp - util_t > 1e-8:
                        # Increase p[t] to make type t prefer truth
                        p[t] = v1t * q1[t] + v2t * q2[t] - max(0, v1t * q1[tp] + v2t * q2[tp] - p[tp])

            # Re-clip
            p = np.clip(p, 0, None)

        allocations = np.column_stack([q1, q2])
        payments = p
    else:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
