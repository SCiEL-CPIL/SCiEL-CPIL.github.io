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

    # Solve with tight tolerance
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': 30, 'mip_rel_gap': 0,
                                  'presolve_mip_gap': 0})
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 2:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if not (result.success and result.x is not None):
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    q1 = np.clip(result.x[0:16], 0, 1)
    q2 = np.clip(result.x[16:32], 0, 1)
    p = np.clip(result.x[32:48], 0, None)

    # Polish: re-solve with tighter xtol if initial solution has small violations
    def check_feasibility(q1, q2, p):
        """Return max violation of IC, IR, nonnegativity, bounds."""
        max_viol = 0.0
        for t in range(n):
            v1t, v2t = types[t]
            util_t = v1t * q1[t] + v2t * q2[t] - p[t]
            # IR: util_t >= 0  =>  p[t] <= v1t*q1[t] + v2t*q2[t]
            if p[t] - v1t * q1[t] - v2t * q2[t] > max_viol:
                max_viol = p[t] - v1t * q1[t] - v2t * q2[t]
            # Nonnegative payment
            if -p[t] > max_viol:
                max_viol = -p[t]
            # Allocation bounds
            if q1[t] < -max_viol:
                max_viol = -q1[t]
            if q1[t] > 1 + max_viol:
                max_viol = q1[t] - 1
            if q2[t] < -max_viol:
                max_viol = -q2[t]
            if q2[t] > 1 + max_viol:
                max_viol = q2[t] - 1
            # IC: for all t' != t, util_t >= util_{t'}
            for tp in range(n):
                if tp == t:
                    continue
                util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                viol = util_tp - util_t
                if viol > max_viol:
                    max_viol = viol
        return max_viol

    viol = check_feasibility(q1, q2, p)
    
    if viol > 1e-8:
        # Try a tighter re-solve with smaller xtol
        try:
            result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                              options={'presolve': True, 'time_limit': 20,
                                       'mip_rel_gap': 0, 'presolve_mip_gap': 0})
            if result2.success and result2.x is not None:
                q1b = np.clip(result2.x[0:16], 0, 1)
                q2b = np.clip(result2.x[16:32], 0, 1)
                pb = np.clip(result2.x[32:48], 0, None)
                viol2 = check_feasibility(q1b, q2b, pb)
                if viol2 < viol:
                    q1, q2, p = q1b, q2b, pb
                    viol = viol2
        except Exception:
            pass

    if viol > 1e-8:
        # Iterative minimal payment adjustment to fix IC violations
        # For each type t, required p[t] >= max_{t' != t} [v1t*q1[t'] + v2t*q2[t'] - p[t'] + (v1t*q1[t] + v2t*q2[t])]
        # But also p[t] <= v1t*q1[t] + v2t*q2[t] (IR) and p[t] >= 0
        for iteration in range(10):
            changed = False
            for t in range(n):
                v1t, v2t = types[t]
                # Minimum payment to satisfy IC for type t
                min_p_ic = 0.0
                for tp in range(n):
                    if tp == t:
                        continue
                    # Need: v1t*q1[t] + v2t*q2[t] - p[t] >= v1t*q1[tp] + v2t*q2[tp] - p[tp]
                    # => p[t] <= v1t*q1[t] + v2t*q2[t] - v1t*q1[tp] - v2t*q2[tp] + p[tp]
                    # Wait, that's an upper bound. Let me re-derive.
                    # IC: u(t, t) >= u(t, t')
                    # v1t*q1[t] + v2t*q2[t] - p[t] >= v1t*q1[tp] + v2t*q2[tp] - p[tp]
                    # => p[t] <= v1t*q1[t] + v2t*q2[t] - v1t*q1[tp] - v2t*q2[tp] + p[tp]
                    # So p[t] has an upper bound from IC, not lower.
                    # This means if IC is violated, we need to DECREASE p[t] or INCREASE p[tp].
                    pass
                
                # Actually, let's just verify and if violated, try to adjust
                util_t = v1t * q1[t] + v2t * q2[t] - p[t]
                max_util_other = -np.inf
                for tp in range(n):
                    if tp == t:
                        continue
                    util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                    if util_tp > max_util_other:
                        max_util_other = util_tp
                
                if max_util_other - util_t > 1e-8:
                    # IC violated: type t would prefer to report tp
                    # We can fix by increasing p[t] won't help (that makes util_t smaller)
                    # We need to either decrease p[t] or increase p[tp]
                    # Decreasing p[t] might violate IR or nonnegativity
                    # Let's try: set p[t] = max(0, v1t*q1[t] + v2t*q2[t] - max_util_other)
                    # This ensures util_t = max_util_other
                    new_p_t = v1t * q1[t] + v2t * q2[t] - max_util_other
                    if new_p_t < p[t]:
                        p[t] = max(0.0, new_p_t)
                        changed = True
            
            if not changed:
                break

        # Final clip
        p = np.clip(p, 0, None)
        q1 = np.clip(q1, 0, 1)
        q2 = np.clip(q2, 0, 1)

    allocations = np.column_stack([q1, q2])
    payments = p

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
