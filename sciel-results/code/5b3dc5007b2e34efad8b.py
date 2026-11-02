"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Scale-invariant, so fix ||h||_1 = 1 and minimize ||h||_2^2 * ||h*h||_inf.
Combines:
  - Precomputed Toeplitz convolution matrix for fast evaluations
  - Multiplicative reweighting (fixed peak) from current program
  - scipy L-BFGS-B local refinement from inspiration 1
  - Differential evolution over compact supports from inspiration 1
  - Diverse structured starts (spikes, blocks, symmetric patterns)
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
TIME_LIMIT = 150.0
SAFETY = 15.0
START = time.time()


def _build_conv_matrix():
    """Build the full autoconvolution Toeplitz matrix C of shape (2N-1, N).
    (C @ h)[k] = sum_i h[i] * h[i - (k - (N-1))]  for k in 0..2N-2."""
    L = 2 * N - 1
    C = np.zeros((L, N))
    for k in range(L):
        shift = k - (N - 1)  # in [-(N-1), N-1]
        for i in range(N):
            j = i - shift
            if 0 <= j < N:
                C[k, i] = 1.0
    return C


C = _build_conv_matrix()
L = 2 * N - 1


def _autoconv(h):
    """Return the full autoconvolution vector of h (length 2N-1)."""
    return C @ h


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    conv = _autoconv(h)
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return 0.0
    return l2sq / (l1 * pinf)


def _cost(h):
    """Scale-invariant cost to minimize: ||h||_2^2 * ||h*h||_inf with ||h||_1=1."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return np.inf
    hn = h / l1
    l2sq = float(hn @ hn)
    conv = _autoconv(hn)
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return np.inf
    return l2sq * pinf


def _neg_score(h):
    """Negative ratio for minimization (scipy)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    conv = _autoconv(h)
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return 0.0
    return -l2sq / (l1 * pinf)


def _reweight(h, peak, iters=200, eps=1e-15):
    """Multiplicative reweighting with fixed peak index."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return h
    h = h / h.sum()
    for _ in range(iters):
        conv = _autoconv(h)
        target = conv[peak]
        if target <= eps:
            break
        l2sq = float(h @ h)
        grad = 2.0 * h * target + l2sq * C[peak, :]
        g = grad - np.mean(grad)
        g = np.maximum(g, 0.0)
        step = 0.5
        h_new = h * np.exp(-step * g / (np.max(g) + eps))
        s = h_new.sum()
        if s <= eps:
            break
        h_new = h_new / s
        h = h_new
    return h


def _lbfgs_refine(h0, bounds, t_end):
    """Local refinement using L-BFGS-B with nonnegativity bounds."""
    h0 = np.array(h0, dtype=float)
    h0 = np.maximum(h0, 0.0)
    if h0.sum() <= 0:
        return h0
    h0 = h0 / h0.sum()

    def obj(x):
        return _neg_score(x)

    def grad(x):
        # Numerical gradient (L-BFGS-B can compute it, but providing it helps)
        # Actually let scipy compute it to save code; use finite differences
        return None

    try:
        res = minimize(
            obj, h0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxls': 50}
        )
        if res.success or res.fun < _neg_score(h0):
            h_new = np.maximum(res.x, 0.0)
            if h_new.sum() > 0:
                h_new = h_new / h_new.sum()
                return h_new
    except Exception:
        pass
    return h0


def _coord_polish(h, peak, rounds=30):
    """1-D line search per coordinate."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return h
    h = h / h.sum()
    best = _cost(h)
    for _ in range(rounds):
        improved = False
        for i in range(N):
            base = h[i]
            mean = h.sum() / N
            cands = [0.0, base, base * 0.5, base * 2.0, mean, 2.0 * mean, 4.0 * mean]
            cands = [c for c in cands if c >= 0.0]
            cur = best
            for c in cands:
                if abs(c - base) < 1e-12:
                    continue
                h2 = h.copy()
                h2[i] = c
                if h2.sum() <= 0:
                    continue
                h2 = h2 / h2.sum()
                cc = _cost(h2)
                if cc < cur - 1e-12:
                    cur = cc
                    h = h2
                    improved = True
        best = cur
        if not improved:
            break
    return h


def _try_start(h0, peak, t_end, bounds):
    """Full pipeline: reweight -> detect peak -> reweight -> L-BFGS -> coord polish."""
    h = _reweight(h0, peak, iters=200)
    if time.time() > t_end:
        return h
    conv = _autoconv(h)
    peak2 = int(np.argmax(np.abs(conv)))
    h = _reweight(h, peak2, iters=100)
    if time.time() > t_end:
        return h
    h = _lbfgs_refine(h, bounds, t_end)
    if time.time() > t_end:
        return h
    h = _coord_polish(h, peak2, rounds=20)
    return h


def _de_optimize(support_size, t_end, seed=42):
    """Differential evolution over the first `support_size` elements."""
    if support_size >= N:
        return None

    def de_obj(x):
        h = np.zeros(N)
        h[:support_size] = x
        return _neg_score(h)

    bounds = [(0.0, 1.0)] * support_size
    try:
        de_res = differential_evolution(
            de_obj, bounds,
            maxiter=100, tol=1e-8, seed=seed,
            polish=True, init='latinhypercube', popsize=12
        )
        if de_res.fun < 0:
            h_full = np.zeros(N)
            h_full[:support_size] = np.maximum(de_res.x, 0.0)
            if h_full.sum() > 0:
                return h_full
    except Exception:
        pass
    return None


def _optimize():
    t_end = START + TIME_LIMIT - SAFETY
    bounds = [(0.0, 1.0)] * N
    rng = np.random.default_rng(12345)

    best = np.ones(N) / N
    best_score = _ratio(best)

    # --- Phase 1: Diverse structured starts with reweight + L-BFGS ---
    starts = []

    # Single spikes
    for s in range(N):
        v = np.zeros(N); v[s] = 1.0
        starts.append(v)

    # Small blocks at various positions
    for w in (2, 3, 4, 5):
        for s in range(0, N - w + 1, max(1, (N - w) // 10)):
            v = np.zeros(N); v[s:s + w] = 1.0 / w
            starts.append(v)

    # Symmetric patterns (mirror around center)
    for k in range(2, 8):
        v = np.zeros(N)
        for i in range(k):
            v[i] = 1.0 / (k + i)
            v[N - 1 - i] = 1.0 / (k + i)
        v = v / v.sum()
        starts.append(v)

    # Asymmetric decaying patterns
    for alpha in (0.3, 0.5, 0.7, 1.0):
        v = np.zeros(N)
        for i in range(N):
            v[i] = np.exp(-alpha * i / N)
        v = v / v.sum()
        starts.append(v)
        v2 = np.zeros(N)
        for i in range(N):
            v2[i] = np.exp(-alpha * (N - 1 - i) / N)
        v2 = v2 / v2.sum()
        starts.append(v2)

    # Random starts
    for _ in range(30):
        v = rng.random(N)
        v[v < 0.4] = 0.0
        if v.sum() > 0:
            v = v / v.sum()
            starts.append(v)

    # Deduplicate
    seen = set()
    uniq = []
    for v in starts:
        key = tuple(np.round(v, 3))
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    starts = uniq

    for idx, v in enumerate(starts):
        if time.time() > t_end:
            break
        conv = _autoconv(v)
        peak = int(np.argmax(np.abs(conv)))
        h = _try_start(v, peak, t_end, bounds)
        sc = _ratio(h)
        if sc > best_score:
            best_score = sc
            best = h.copy()

        # Try alternative peak
        alt = N - 1 - peak
        if alt != peak and time.time() < t_end:
            h2 = _try_start(v, alt, t_end, bounds)
            sc2 = _ratio(h2)
            if sc2 > best_score:
                best_score = sc2
                best = h2.copy()

    # --- Phase 2: Differential evolution over compact supports ---
    for support_size in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > t_end:
            break
        for seed in (42, 123, 777):
            if time.time() > t_end:
                break
            h_de = _de_optimize(support_size, t_end, seed=seed)
            if h_de is not None:
                # Refine with L-BFGS
                h_de = _lbfgs_refine(h_de, bounds, t_end)
                h_de = _coord_polish(h_de, int(np.argmax(np.abs(_autoconv(h_de)))), rounds=15)
                sc = _ratio(h_de)
                if sc > best_score:
                    best_score = sc
                    best = h_de.copy()

    # --- Phase 3: Final L-BFGS polish from best ---
    if time.time() < t_end:
        peak = int(np.argmax(np.abs(_autoconv(best))))
        best = _lbfgs_refine(best, bounds, t_end)
        best = _coord_polish(best, peak, rounds=20)
        best_score = _ratio(best)

    # Ensure nonnegativity and normalization
    best = np.maximum(best, 0.0)
    if best.sum() > 0:
        best = best / best.sum()

    return best, best_score


_cache = None


def construct():
    global _cache
    if _cache is None:
        _cache = _optimize()
    best, _ = _cache
    return [float(x) for x in best]
