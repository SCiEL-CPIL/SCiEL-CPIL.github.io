"""Incentive-compatible finite auction design (one additive bidder, two goods)."""
import time
import numpy as np
from itertools import product

_DEADLINE = 85.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _time_ok(margin=3.0):
    return _elapsed() < _DEADLINE - margin


def _lp_optimum(p, v1, v2):
    """Independently solved finite-support LP: max E[pay] s.t. BIC + IR + 0<=x<=1.

    Variables: x_{v1,v2} in [0,1], pay_{v1,v2} >= 0.
    BIC: pay_{v1,v2} - pay_{v1',v2'} - (v1-v1')x_{v1',v2} - (v2-v2')x_{v1,v2'} <= 0.
    IR:  pay_{v1,v2} - v1*x_{v1,v2} - v2*x_{v1,v2} <= 0.
    """
    from scipy.optimize import linprog
    V = list(range(4))
    idx = {(a, b): i for i, (a, b) in enumerate(product(V, V))}
    n = 16
    c = np.zeros(2 * n)
    for (a, b) in product(V, V):
        c[n + idx[(a, b)]] = -p[a, b]
    A_ub = []
    b_ub = []
    # BIC
    for (a, b) in product(V, V):
        for (a2, b2) in product(V, V):
            if (a, b) == (a2, b2):
                continue
            row = np.zeros(2 * n)
            row[n + idx[(a, b)]] = 1.0
            row[n + idx[(a2, b2)]] = -1.0
            row[idx[(a2, b)]] = (a - a2)
            row[idx[(a, b2)]] = (b - b2)
            A_ub.append(row)
            b_ub.append(0.0)
    # IR
    for (a, b) in product(V, V):
        row = np.zeros(2 * n)
        row[n + idx[(a, b)]] = 1.0
        row[idx[(a, b)]] = -(a + b)
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, 1.0)] * n + [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        return max(0.0, float(res.fun))
    return 1.0


def _eval_mech(p, v1, v2, alloc, pay, support_mask):
    """Return (expected_revenue, feasible) for a 2D allocation/payment table."""
    # feasibility: probabilities in [0,1]
    if np.any(alloc < -1e-9) or np.any(alloc > 1.0 + 1e-9):
        return 0.0, False
    # nonnegative payments
    if np.any(pay < -1e-9):
        return 0.0, False
    # individual rationality (on support)
    for (a, b) in support_mask:
        if pay[a, b] - (a + b) * alloc[a, b] > 1e-8:
            return 0.0, False
    # truthfulness / regret <= 1e-8 (on support)
    for (a, b) in support_mask:
        u = pay[a, b] - (a + b) * alloc[a, b]
        for (a2, b2) in support_mask:
            if (a2, b2) == (a, b):
                continue
            dev = pay[a2, b2] - (a + b) * alloc[a2, b2]
            if dev - u > 1e-8:
                return 0.0, False
    # expected revenue
    rev = 0.0
    for (a, b) in support_mask:
        rev += p[a, b] * pay[a, b]
    return rev, True


def _threshold_mech(th1, th2):
    """Build allocation/payment for a 2D threshold rule.

    x(v1,v2)=1 iff v1>=th1 AND v2>=th2.
    pay(v1,v2) = (v1-th1)*x + (v2-th2)*x  (Myerson payment, nonneg via x).
    """
    alloc = np.zeros((4, 4))
    pay = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            x = 1.0 if (a >= th1 and b >= th2) else 0.0
            alloc[a, b] = x
            pay[a, b] = (a - th1) * x + (b - th2) * x
    return alloc, pay


def construct_auction(distribution_spec, support, constraints):
    global _START
    _START = time.time()

    # ---- Parse distribution ----
    p = None
    if isinstance(distribution_spec, dict):
        for k in ("prob", "probs", "distribution", "pdf", "weights", "mass", "P", "p"):
            if k in distribution_spec:
                p = np.asarray(distribution_spec[k], dtype=float)
                break
        if p is None:
            # maybe a 2D array under a key
            for k, v in distribution_spec.items():
                try:
                    arr = np.asarray(v, dtype=float)
                    if arr.ndim == 2 and arr.shape == (4, 4):
                        p = arr
                        break
                except Exception:
                    pass
    if p is None:
        p = np.asarray(distribution_spec, dtype=float)
    if p.ndim != 2 or p.shape != (4, 4):
        # try flatten 16
        pf = np.asarray(distribution_spec, dtype=float).ravel()
        if pf.size == 16:
            p = pf.reshape(4, 4)
        else:
            p = np.full((4, 4), 1.0 / 16.0)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s > 0:
        p = p / s

    # ---- Support mask ----
    if support is None:
        support_mask = [(a, b) for a in range(4) for b in range(4)]
    else:
        sm = set()
        try:
            for item in support:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    sm.add((int(item[0]), int(item[1])))
                elif isinstance(item, (list, tuple)) and len(item) == 1:
                    sm.add((int(item[0]), int(item[0])))
        except Exception:
            sm = set()
        if not sm:
            support_mask = [(a, b) for a in range(4) for b in range(4)]
        else:
            support_mask = list(sm)

    # ---- Independently solved LP optimum (normalizer) ----
    try:
        lp_opt = _lp_optimum(p, 0, 0)
    except Exception:
        lp_opt = 1.0
    if lp_opt < 1e-9:
        lp_opt = 1.0

    # ---- Enumerate threshold combinations ----
    best_revenue = -1.0
    best_alloc = None
    best_pay = None
    candidates = [(t1, t2) for t1 in range(4) for t2 in range(4)]
    # Also consider "no-good" / mixed single-good thresholds are covered by 2D rule
    for (t1, t2) in candidates:
        if not _time_ok():
            break
        alloc, pay = _threshold_mech(t1, t2)
        rev, feas = _eval_mech(p, 0, 0, alloc, pay, support_mask)
        if feas and rev > best_revenue:
            best_revenue = rev
            best_alloc = alloc.copy()
            best_pay = pay.copy()

    # Fallback: deterministic full-allocation with zero payment (IR, truthful? check)
    if best_alloc is None:
        alloc = np.ones((4, 4))
        pay = np.zeros((4, 4))
        rev, feas = _eval_mech(p, 0, 0, alloc, pay, support_mask)
        if feas:
            best_revenue = rev
            best_alloc = alloc
            best_pay = pay
        else:
            # zero allocation, zero payment is always feasible
            best_alloc = np.zeros((4, 4))
            best_pay = np.zeros((4, 4))
            best_revenue = 0.0

    # ---- Reshape to required output: (16,2) allocations, (16,) payments ----
    # Order: enumerate (v1 in 0..3, v2 in 0..3) row-major
    alloc_flat = np.zeros((16, 2))
    pay_flat = np.zeros(16)
    i = 0
    for a in range(4):
        for b in range(4):
            alloc_flat[i, 0] = best_alloc[a, b]
            alloc_flat[i, 1] = best_alloc[a, b]
            pay_flat[i] = best_pay[a, b]
            i += 1

    # Safety margin: ensure we return well before deadline
    if _elapsed() > _DEADLINE - 5.0:
        pass  # already near limit; return immediately

    return {"allocations": alloc_flat, "payments": pay_flat}
