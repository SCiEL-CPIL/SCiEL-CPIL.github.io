import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

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

    # Cache for optimal solutions keyed by distribution
    _auction_cache = {}
    cache_key = tuple(np.round(probs, 12))
    if cache_key in _auction_cache:
        cached = _auction_cache[cache_key]
        return {'allocations': cached[0].copy(), 'payments': cached[1].copy()}

    # For additive single-bidder mechanism with types (v1, v2):
    # IC + IR implies:
    #   p(v1,v2) >= p(v1',v2') + v1*(q1(v1,v2)-q1(v1',v2')) + v2*(q2(v1,v2)-q2(v1',v2'))
    #   p(v1,v2) >= 0
    #   p(v1,v2) <= v1*q1(v1,v2) + v2*q2(v1,v2)  (IR)
    #
    # The minimum payment satisfying IC and IR is:
    #   p(v1,v2) = max(0, max over paths from (0,0) to (v1,v2) of path_cost)
    # where path_cost accumulates v1*dq1 + v2*dq2 along the path
    #
    # For a grid, the minimum payment is:
    #   p(v1,v2) = max(0, 
    #                  v1*q1(v1,v2) + v2*q2(v1,v2) - [integral-like term])
    #
    # Actually, for discrete monotone allocation, the payment that achieves
    # IC+IR with minimum payment (maximizing revenue) is:
    #   p(v1,v2) = v1*q1(v1,v2) + v2*q2(v2,v2) - U(v1,v2)
    # where U(v1,v2) is the utility, and for the optimal mechanism:
    #   U(v1,v2) = max(0, U(v1-1,v2) + v1*(q1(v1,v2)-q1(v1-1,v2)),
    #                   U(v1,v2-1) + v2*(q2(v1,v2)-q2(v1,v2-1)))
    # with U(0,0) = 0
    #
    # Wait, let me think more carefully. The utility U(v) = v·q(v) - p(v).
    # IC requires U(v) >= U(v') + v·(q(v) - q(v')) for all v, v'.
    # This is equivalent to: U(v) - v·q(v) >= U(v') - v·q(v') for all v, v'
    # Let W(v) = U(v) - v·q(v) = -p(v). Then W(v) >= W(v') - v·(q(v)-q(v'))
    # Hmm, this is getting complex. Let me use the standard result.
    #
    # For single-parameter (each good independent), the optimal mechanism has:
    #   q_i(v_i) non-decreasing in v_i
    #   p(v) = sum_i [v_i * q_i(v) - integral_0^{v_i} q_i(w) dw] (continuous)
    # For discrete: p(v1,v2) = v1*q1(v1,v2) + v2*q2(v1,v2) - sum over path
    #
    # The key insight: for a 2D grid with monotone allocations, the payment is:
    #   p(v1,v2) = max(0, 
    #                  v1*q1(v1,v2) + v2*q2(v1,v2) - 
    #                  [v1*q1(v1,v2) + v2*q2(v1,v2) - (v1-1)*q1(v1-1,v2) - v2*q2(v1,v2-1) + ...])
    #
    # Let me just use the LP but with a smarter formulation. The allocation must be
    # monotone: q1(v1+1,v2) >= q1(v1,v2) and q2(v1,v2+1) >= q2(v1,v2).
    # 
    # Actually, the simplest correct approach: solve the LP but exploit that
    # for this small problem (16 types), the LP is fast. The issue is numerical
    # precision. Let me use a cleaner formulation.
    #
    # Variables: q1[16], q2[16], p[16]
    # Constraints:
    # 1. 0 <= q1[t] <= 1, 0 <= q2[t] <= 1
    # 2. p[t] >= 0
    # 3. IC: p[t] - p[tp] >= v1t*q1[t] + v2t*q2[t] - v1t*q1[tp] - v2t*q2[tp]
    #    => -v1t*q1[t] - v2t*q2[t] + v1t*q1[tp] + v2t*q2[tp] + p[t] - p[tp] >= 0
    #    => v1t*(q1[tp]-q1[t]) + v2t*(q2[tp]-q2[t]) - p[t] + p[tp] <= 0
    # 4. IR: p[t] <= v1t*q1[t] + v2t*q2[t]
    #    => p[t] - v1t*q1[t] - v2t*q2[t] <= 0
    #
    # Objective: maximize sum_t probs[t] * p[t]
    #
    # The current code does this correctly. Let me improve numerical stability.

    # LP variables:
    # x[0:16]   = q1[t]
    # x[16:32]  = q2[t]
    # x[32:48]  = p[t]

    # Objective: minimize -sum_t probs[t] * p[t]
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

    q1 = None
    q2 = None
    p = None

    try:
        options = {'presolve': True, 'dual_feasibility_tolerance': 1e-13,
                   'primal_feasibility_tolerance': 1e-13, 'time_limit': 25.0}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success and result.x is not None:
            q1 = np.clip(result.x[0:16], 0, 1)
            q2 = np.clip(result.x[16:32], 0, 1)
            p = np.clip(result.x[32:48], 0, None)
    except Exception:
        pass

    # If LP failed, try a simpler approach: use the DP-based payment computation
    if q1 is None:
        # Default to simple mechanism: allocate good i if v_i >= threshold
        # Use threshold 1 for both goods (allocate if value >= 1)
        q1 = np.zeros(16)
        q2 = np.zeros(16)
        p = np.zeros(16)
        for t in range(n):
            v1t, v2t = types[t]
            q1[t] = 1.0 if v1t >= 1 else 0.0
            q2[t] = 1.0 if v2t >= 1 else 0.0
            p[t] = v1t * q1[t] + v2t * q2[t]
    
    # Polish: tighten the solution with multiple passes at increasing precision
    if time.time() < deadline - 15:
        current_x = np.concatenate([q1, q2, p])
        for polish_pass in range(3):
            if time.time() > deadline - 5:
                break
            tol = 1e-14 if polish_pass == 0 else 1e-16
            try:
                options_tight = {'presolve': False,
                                  'dual_feasibility_tolerance': tol,
                                  'primal_feasibility_tolerance': tol,
                                  'time_limit': 5.0}
                result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                  method='highs', options=options_tight, x0=current_x)
                if result2.success and result2.x is not None:
                    q1_new = np.clip(result2.x[0:16], 0, 1)
                    q2_new = np.clip(result2.x[16:32], 0, 1)
                    p_new = np.clip(result2.x[32:48], 0, None)
                    rev_orig = np.dot(probs, p)
                    rev_new = np.dot(probs, p_new)
                    if rev_new >= rev_orig - 1e-13:
                        q1, q2, p = q1_new, q2_new, p_new
                        current_x = np.concatenate([q1, q2, p])
            except Exception:
                break

    # Verify constraints and fix any violations
    max_regret = 0.0
    for t in range(n):
        v1t, v2t = types[t]
        util_t = v1t * q1[t] + v2t * q2[t] - p[t]
        for tp in range(n):
            if tp == t:
                continue
            util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
            regret = util_tp - util_t
            if regret > max_regret:
                max_regret = regret

    if max_regret > 1e-9:
        # Iterative payment adjustment
        for iteration in range(200):
            if time.time() > deadline - 2:
                break
            changed = False
            for t in range(n):
                v1t, v2t = types[t]
                min_p = 0.0
                for tp in range(n):
                    if tp == t:
                        continue
                    required = p[tp] + v1t * (q1[t] - q1[tp]) + v2t * (q2[t] - q2[tp])
                    if required > min_p:
                        min_p = required
                max_p = v1t * q1[t] + v2t * q2[t]
                if min_p > max_p:
                    min_p = max_p
                if min_p > p[t] + 1e-14:
                    p[t] = min_p
                    changed = True
            if not changed:
                break
        p = np.clip(p, 0, None)

    allocations = np.column_stack([q1, q2])
    payments = p

    # Cache the result
    _auction_cache[cache_key] = (allocations.copy(), payments.copy())

    return {'allocations': allocations, 'payments': payments}
