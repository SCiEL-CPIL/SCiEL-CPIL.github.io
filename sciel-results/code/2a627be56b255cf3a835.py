# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _get_probs(distribution_spec, n_types):
    """Extract type probabilities from distribution_spec (robust to several formats)."""
    if distribution_spec is None:
        return np.full(n_types, 1.0 / n_types)
    if isinstance(distribution_spec, dict):
        for key in ('probs', 'probabilities', 'p', 'weights', 'dist'):
            if key in distribution_spec:
                arr = np.asarray(distribution_spec[key], dtype=float).ravel()
                if arr.size == n_types:
                    s = arr.sum()
                    if s > 0:
                        return arr / s
                    return arr
        # maybe nested per-goods
        v1 = distribution_spec.get('v1', distribution_spec.get('goods0', None))
        v2 = distribution_spec.get('v2', distribution_spec.get('goods1', None))
        if v1 is not None and v2 is not None:
            p1 = np.asarray(v1, dtype=float).ravel()
            p2 = np.asarray(v2, dtype=float).ravel()
            if p1.size == 4 and p2.size == 4:
                grid = np.outer(p1, p2)
                return grid.ravel()
        # fallback: any array of length 16
        for v in distribution_spec.values():
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == n_types:
                s = arr.sum()
                if s > 0:
                    return arr / s
    arr = np.asarray(distribution_spec, dtype=float).ravel()
    if arr.size == n_types:
        s = arr.sum()
        if s > 0:
            return arr / s
    return np.full(n_types, 1.0 / n_types)


def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s

    # Build type list: all (v1,v2) in {0,1,2,3}^2, in a fixed canonical order.
    vals = [0, 1, 2, 3]
    types = [(a, b) for a in vals for b in vals]
    n = len(types)  # 16
    assert n == 16

    # If support is provided and matches, use it to align ordering; otherwise canonical.
    if support is not None:
        sup = list(support)
        if len(sup) == n:
            # map to canonical value grid
            types = [tuple(int(x) for x in t) for t in sup]

    probs = _get_probs(distribution_spec, n)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        probs = np.full(n, 1.0 / n)
    else:
        probs = probs / s

    V = np.array([[v1, v2] for (v1, v2) in types], dtype=float)  # (16,2)

    # LP: maximize expected payment.
    # vars: a_{t,1}, a_{t,2} for t=0..15, then p_t for t=0..15.
    # index: 0..15 -> a1, 16..31 -> a2, 32..47 -> p
    nA = 2 * n  # 32
    nP = n      # 16
    nvar = nA + nP  # 48

    def a1_idx(t): return t
    def a2_idx(t): return n + t
    def p_idx(t): return nA + t

    c = np.zeros(nvar)
    for t in range(n):
        c[p_idx(t)] = -probs[t]  # minimize -expected payment

    # Inequality constraints A_ub x <= b_ub
    A_rows = []
    b_rows = []

    # IC: for all t (true), t' (report):
    # v(t).a(t) - p_t >= v(t).a(t') - p_t'
    # => v(t).a(t') - v(t).a(t) + p_t - p_t' <= 0
    for t in range(n):
        for tp in range(n):
            if t == tp:
                continue
            row = np.zeros(nvar)
            # v(t).a(t')
            row[a1_idx(tp)] += V[t, 0]
            row[a2_idx(tp)] += V[t, 1]
            # - v(t).a(t)
            row[a1_idx(t)] -= V[t, 0]
            row[a2_idx(t)] -= V[t, 1]
            # + p_t
            row[p_idx(t)] += 1.0
            # - p_t'
            row[p_idx(tp)] -= 1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # IR: v(t).a(t) - p_t >= 0  =>  p_t - v(t).a(t) <= 0
    for t in range(n):
        row = np.zeros(nvar)
        row[p_idx(t)] += 1.0
        row[a1_idx(t)] -= V[t, 0]
        row[a2_idx(t)] -= V[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows, dtype=float)
    b_ub = np.array(b_rows, dtype=float)

    # Bounds: a in [0,1], p in [0, inf]
    bounds = []
    for i in range(nA):
        bounds.append((0.0, 1.0))
    for i in range(nP):
        bounds.append((0.0, None))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        # Fallback: deterministic Myerson-like — allocate both goods to all types, payment 0 (IR, IC trivially? no)
        # Safer fallback: allocate nothing, pay 0 (feasible, truthful, IR, revenue 0)
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    x = res.x
    allocations = np.zeros((n, 2))
    payments = np.zeros(n)
    for t in range(n):
        allocations[t, 0] = np.clip(x[a1_idx(t)], 0.0, 1.0)
        allocations[t, 1] = np.clip(x[a2_idx(t)], 0.0, 1.0)
        payments[t] = np.clip(x[p_idx(t)], 0.0, None)

    # Polish: verify constraints; if tiny violations, project payments minimally.
    # Check IC and IR; if violated beyond tolerance, adjust payments up to satisfy (raising payment preserves IR? no, raising payment hurts IR).
    # Instead, if violated, fall back to a safe monotone deterministic mechanism.
    tol = 1e-8
    ok = True
    for t in range(n):
        util_t = V[t] @ allocations[t] - payments[t]
        if util_t < -tol:
            ok = False
            break
        for tp in range(n):
            util_tp_as_t = V[t] @ allocations[tp] - payments[tp]
            if util_t < util_tp_as_t - tol:
                ok = False
                break
        if not ok:
            break

    if not ok:
        # Fallback: allocate both goods to every type, payment = max value (IR, but IC? payment constant => IC holds since allocation constant)
        allocations = np.ones((n, 2))
        payments = np.array([V[t] @ [1, 1] for t in range(n)])  # payment = value, util 0, IR ok, IC: alloc constant so IC holds
        # Check: v(t).a(t)-p_t = 0 for all; v(t).a(t')-p_t' = v(t).2 - (v1'+v2'). Not necessarily <= 0.
        # Constant allocation => IC requires v(t).a - p_t >= v(t).a - p_t' => p_t <= p_t' for all t,t' => all payments equal.
        # So set payment = constant = max total value to keep IR? IR: v(t).2 - p >=0 => p <= 2*min v =0. So p=0, util=2v(t)>=0 IR ok. IC: p equal => holds.
        allocations = np.ones((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
