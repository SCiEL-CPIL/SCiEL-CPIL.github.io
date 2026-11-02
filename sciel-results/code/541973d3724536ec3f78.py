import numpy as np
import time
import warnings
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _score(h):
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0 or np.any(h < 0):
        return 0.0
    h = h / s
    l2sq = float(np.sum(h * h))
    conv = _autoconv(h)
    m = float(np.max(conv))
    if m <= 0:
        return 0.0
    return l2sq / m

def _neg_score(h):
    s = h.sum()
    if s <= 0 or np.any(h < 0):
        return 1e6
    h = h / s
    l2sq = float(np.sum(h * h))
    conv = _autoconv(h)
    m = float(np.max(conv))
    if m <= 0:
        return 1e6
    return -(l2sq / m)

def _sidon_supports(max_m=10, seed=0):
    rng = np.random.RandomState(seed)
    supports = []
    for m in range(2, max_m + 1):
        sup = np.sort(rng.choice(N, size=m, replace=False))
        supports.append(sup)
    for start in range(0, N - 6, 3):
        supports.append(np.arange(start, start + 8))
    g = np.array([0, 1, 3, 7, 15, 31])
    supports.append(g[g < N])
    supports.append(np.array([0, 1, 4, 10, 21, 43] if 43 < N else [0, 1, 4, 10, 21]))
    # More structured: Bose-Chowla style
    for base in [2, 3, 4]:
        sup = []
        x = 0
        while x < N and len(sup) < 12:
            sup.append(x)
            x = x * base + 1
        if len(sup) >= 2:
            supports.append(np.array(sup[:12]))
    return supports

def _slsqp_optimize(h0, peak_bias=None, maxiter=200, tol=1e-8, deadline=None):
    """Optimize using SLSQP on the simplex. peak_bias: if given, add a penalty
    for autoconv peaks exceeding the value at peak_bias lag."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h0.sum()
    if s <= 0:
        h0 = np.ones(N) / N
    else:
        h0 = h0 / s

    def objective(x):
        if deadline is not None and time.time() > deadline:
            return 1e6
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s <= 0:
            return 1e6
        x = x / s
        val = _neg_score(x)
        if peak_bias is not None and 0 <= peak_bias < len(_autoconv(x)):
            conv = _autoconv(x)
            target = conv[peak_bias]
            if target > 0:
                excess = np.maximum(conv - target, 0.0)
                val += 0.1 * float(np.sum(excess * excess)) / (target * target + 1e-12)
        return val

    constraints = [
        {'type': 'eq', 'fun': lambda x: x.sum() - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N

    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': maxiter, 'ftol': tol})
        if res.success or res.fun < 1e5:
            h = np.maximum(res.x, 0.0)
            s = h.sum()
            if s > 0:
                h = h / s
                return h
    except Exception:
        pass
    return h0

def _coordinate_polish(h, n_passes=200, tol=1e-12):
    h = np.asarray(h, dtype=float).copy()
    s = h.sum()
    if s <= 0:
        return h
    best = _score(h)
    for _ in range(n_passes):
        improved = False
        for i in range(N):
            if h[i] < 1e-15 and best > 1.15:
                continue
            others = h.copy()
            others[i] = 0.0
            base_sum = others.sum()
            cands = [h[i]]
            if h[i] > 0:
                cands += [h[i] * 0.5, h[i] * 1.5, h[i] * 0.1, h[i] * 3.0, 0.0]
            else:
                cands += [base_sum * 0.01, base_sum * 0.05, base_sum * 0.2]
            cands = [max(c, 0.0) for c in cands]
            cur_best = best
            cur_val = h[i]
            for c in cands:
                test = others.copy()
                test[i] = c
                sc = _score(test)
                if sc > cur_best + 1e-12:
                    cur_best = sc
                    cur_val = c
                    improved = True
            h[i] = cur_val
        if not improved:
            break
    return h

def construct():
    DEADLINE = time.time() + 135
    best_h = None
    best_score = -1.0

    def consider(h):
        nonlocal best_h, best_score
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        s = h.sum()
        if s <= 0:
            return
        h = h / s
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 1) Quick diverse starts
    rng = np.random.RandomState(42)
    starts = []
    for a in [0.8, 0.9, 0.95, 0.7, 0.6]:
        v = np.zeros(N)
        v[0] = a
        v[1] = 1 - a
        starts.append(v)
    for a, b in [(0.6, 0.3), (0.7, 0.2), (0.5, 0.4), (0.8, 0.1)]:
        v = np.zeros(N)
        v[0] = a
        v[1] = b
        v[2] = 1 - a - b
        if v[2] >= 0:
            starts.append(v)
    for _ in range(15):
        m = rng.randint(2, 9)
        sup = np.sort(rng.choice(N, size=m, replace=False))
        v = np.zeros(N)
        w = rng.rand(m)
        v[sup] = w / w.sum()
        starts.append(v)
    for _ in range(5):
        v = rng.rand(N)
        v = v / v.sum()
        starts.append(v)
    for r in [0.5, 0.6, 0.7, 0.8]:
        v = np.zeros(N)
        val = 1.0
        for i in range(12):
            v[i] = val
            val *= r
        v = v / v.sum()
        starts.append(v)

    for v in starts:
        consider(v)
    if best_h is None:
        best_h = np.zeros(N); best_h[0] = 1.0

    # 2) SLSQP optimization from diverse starts
    supports = _sidon_supports(max_m=10, seed=1)
    opt_starts = []
    for v in starts[:20]:
        opt_starts.append(v)
    for sup in supports:
        m = len(sup)
        v = np.zeros(N)
        v[sup] = 1.0 / m
        opt_starts.append(v)
    # Add a few more random
    for _ in range(10):
        m = rng.randint(2, 10)
        sup = np.sort(rng.choice(N, size=m, replace=False))
        v = np.zeros(N)
        w = rng.rand(m)
        v[sup] = w / w.sum()
        opt_starts.append(v)

    # Deduplicate roughly
    seen = []
    unique_starts = []
    for v in opt_starts:
        v = np.maximum(v, 0.0)
        s = v.sum()
        if s <= 0:
            continue
        v = v / s
        dup = False
        for u in seen:
            if np.sum(np.abs(v - u)) < 0.1:
                dup = True
                break
        if not dup:
            seen.append(v)
            unique_starts.append(v)

    # Run SLSQP on a subset with time budget
    n_opt = min(len(unique_starts), 25)
    for i in range(n_opt):
        if time.time() > DEADLINE - 30:
            break
        h0 = unique_starts[i]
        # Try with peak_bias = None first
        h1 = _slsqp_optimize(h0, peak_bias=None, maxiter=150, deadline=DEADLINE - 5)
        consider(h1)
        # Try with a few peak biases
        if time.time() < DEADLINE - 20:
            conv = _autoconv(h1)
            peak_idx = int(np.argmax(conv))
            h2 = _slsqp_optimize(h1, peak_bias=peak_idx, maxiter=100, deadline=DEADLINE - 5)
            consider(h2)

    # 3) Coordinate polish the global best
    if time.time() < DEADLINE - 15:
        polished = _coordinate_polish(best_h, n_passes=300)
        consider(polished)

    # 4) SLSQP from the polished best
    if time.time() < DEADLINE - 10:
        h3 = _slsqp_optimize(best_h, peak_bias=None, maxiter=200, deadline=DEADLINE - 3)
        consider(h3)

    # 5) Final polish
    if time.time() < DEADLINE - 5:
        polished2 = _coordinate_polish(best_h, n_passes=200)
        consider(polished2)

    best_h = np.maximum(best_h, 0.0)
    s = best_h.sum()
    if s <= 0:
        best_h = np.zeros(N); best_h[0] = 1.0
    else:
        best_h = best_h / s
    return best_h
# EVOLVE-BLOCK-END
