"""Erdos minimum overlap: minimize max autocorrelation of h vs 1-h."""
import time

N = 100
DEADLINE = 150.0
START = time.time()
TIME_MARGIN = 8.0  # safety margin before the 150s wall-clock kill


def remaining_time():
    return DEADLINE - TIME_MARGIN - (time.time() - START)


def _value(h):
    """The evaluator's raw objective (lower is better)."""
    import numpy as np
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _valid(h):
    import numpy as np
    h = np.asarray(h, dtype=float)
    if h.shape != (N,) or not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-8) or np.any(h > 1.0 + 1e-8):
        return False
    if abs(float(np.sum(h)) - N / 2.0) > 1e-9 * N:
        return False
    return True


def _project(h):
    """Project onto {0<=h<=1, sum(h)=N/2} via bisection on the box threshold."""
    import numpy as np
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    target = N / 2.0
    s = float(np.sum(h))
    if s <= target:
        return h
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(h - mid, 0.0, 1.0))) >= target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return np.clip(h - mid, 0.0, 1.0)


def _solve_lp():
    """Exact LP: min t s.t. autocorr(k)<=t for all k, 0<=h<=1, sum(h)=N/2."""
    import numpy as np
    from scipy.optimize import linprog

    c = np.zeros(N + 1)
    c[N] = 1.0  # minimize t

    # Inequalities: autocorr(k) <= t  ->  A_ineq @ x <= b
    A_rows = []
    b_rows = []
    for k in range(-N + 1, N):
        idx = k + N - 1
        row = np.zeros(N + 1)
        for j in range(N):
            jk = j + k
            if 0 <= jk < N:
                row[j] += 1.0
                row[jk] -= 1.0
        row[N] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
    A_ineq = np.array(A_rows)
    b_ineq = np.array(b_rows)

    # Equality: sum(h) = N/2
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([N / 2.0])

    bounds = [(0.0, 1.0)] * N + [(None, None)]
    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None
    h = res.x[:N]
    h = _project(h)
    return h


def _polish(h0, iters=2000, step=0.02):
    """Polish with projected gradient descent on the smooth max surrogate."""
    import numpy as np
    h = np.clip(np.asarray(h0, dtype=float), 0.0, 1.0)
    best = h.copy()
    best_v = _value(best)
    for it in range(iters):
        if remaining_time() < 1.5:
            break
        corr = np.correlate(h, 1.0 - h, mode="full")
        grad = np.zeros(N)
        for k in range(-N + 1, N):
            idx = k + N - 1
            for j in range(N):
                jk = j + k
                if 0 <= jk < N:
                    grad[j] += corr[idx]
                    grad[jk] -= corr[idx]
        h = _project(h - step * grad)
        v = _value(h)
        if v < best_v:
            best_v = v
            best = h.copy()
    return best, best_v


def construct():
    import numpy as np

    best = np.full(N, 0.5)
    best_v = _value(best)

    # Phase 1: exact LP solution (the core of the design).
    if remaining_time() > 3.0:
        h = _solve_lp()
        if h is not None and _valid(h):
            v = _value(h)
            if v < best_v:
                best, best_v = h, v

    # Phase 2: high-precision polish (tighter effective tolerance).
    if remaining_time() > 2.0:
        h, v = _polish(best, iters=4000, step=0.02)
        if _valid(h) and v < best_v:
            best, best_v = h, v

    # Final safety: guarantee a valid, sum-corrected output.
    best = _project(best)
    return best
