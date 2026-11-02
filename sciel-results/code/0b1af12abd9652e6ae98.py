"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Scale-invariant, so fix ||h||_1 = 1 and minimize ||h||_2^2 * ||h*h||_inf.

Structural approach:
  - Precomputed Toeplitz convolution matrix
  - Peak-tracking exact coordinate descent
  - Simulated annealing with adaptive step sizes
  - Multi-start with diverse structured + random initializations
  - Final intensive refinement
"""

import time
import numpy as np

N = 50
TIME_LIMIT = 150.0
SAFETY = 15.0
START = time.time()


def _build_conv_matrix():
    """Build the full autoconvolution Toeplitz matrix C of shape (2N-1, N)."""
    L = 2 * N - 1
    C = np.zeros((L, N))
    for k in range(L):
        shift = k - (N - 1)
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


def _exact_coord_search(h, i, peak, tol=1e-10):
    """Exact 1-D search for coordinate i given fixed peak bin."""
    a = h[i]
    S = float(h @ h) - a * a
    conv = _autoconv(h)
    T = conv[peak] - C[peak, i] * a
    ci = C[peak, i]
    
    def f(delta):
        d = 1.0 + delta
        if d <= 1e-15:
            return np.inf
        l2n = (S + (a + delta) ** 2) / (d * d)
        cv = (T + ci * (a + delta)) / d
        if cv <= 0:
            return np.inf
        return l2n * cv
    
    lo = -(a - 1e-12)
    hi = max(10.0 * a + 1.0, 0.1)
    
    f_lo = f(lo)
    f_hi = f(hi)
    
    gr = (np.sqrt(5) - 1) / 2
    a1 = hi - gr * (hi - lo)
    a2 = lo + gr * (hi - lo)
    f_a1 = f(a1)
    f_a2 = f(a2)
    
    for _ in range(60):
        if f_a1 < f_a2:
            hi = a2
            a2 = a1
            f_a2 = f_a1
            a1 = hi - gr * (hi - lo)
            f_a1 = f(a1)
        else:
            lo = a1
            a1 = a2
            f_a1 = f_a2
            a2 = lo + gr * (hi - lo)
            f_a2 = f(a2)
    
    best_delta = (lo + hi) / 2
    best_val = f(best_delta)
    
    f_zero = f(-a)
    if f_zero < best_val:
        best_delta = -a
        best_val = f_zero
    
    new_a = a + best_delta
    if new_a < 0:
        new_a = 0.0
    h_new = h.copy()
    h_new[i] = new_a
    if h_new.sum() > 0:
        h_new = h_new / h_new.sum()
    return h_new, best_val


def _peak_tracking_descent(h, max_rounds=50):
    """Coordinate descent that tracks which conv bin is the peak."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return h
    h = h / h.sum()
    best_cost = _cost(h)
    
    for rnd in range(max_rounds):
        conv = _autoconv(h)
        peak = int(np.argmax(conv))
        improved = False
        for i in range(N):
            h_new, new_cost = _exact_coord_search(h, i, peak)
            conv_new = _autoconv(h_new)
            if conv_new[peak] >= np.max(conv_new) - 1e-12:
                if new_cost < best_cost - 1e-14:
                    h = h_new
                    best_cost = new_cost
                    improved = True
            else:
                actual = _cost(h_new)
                if actual < best_cost - 1e-14:
                    h = h_new
                    best_cost = actual
                    improved = True
        if not improved:
            break
    return h


def _anneal(h, t_end, n_iter=2000):
    """Simulated annealing with adaptive step sizes."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return h
    h = h / h.sum()
    best_h = h.copy()
    best_cost = _cost(h)
    cur_cost = best_cost
    rng = np.random.default_rng(42)
    
    t_start = time.time()
    T_init = 0.01
    T_final = 1e-6
    step_scale = 1.0
    
    for it in range(n_iter):
        if time.time() > t_end:
            break
        T = T_init * (T_final / T_init) ** (it / n_iter)
        
        n_pert = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        h_new = h.copy()
        for i in idxs:
            scale = 0.5 + rng.random() * 1.5
            h_new[i] = max(0.0, h[i] * scale + rng.normal(0, 0.05 * step_scale))
        
        if h_new.sum() <= 0:
            continue
        h_new = h_new / h_new.sum()
        new_cost = _cost(h_new)
        
        if new_cost < cur_cost or rng.random() < np.exp(-(new_cost - cur_cost) / (T + 1e-15)):
            h = h_new
            cur_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
                step_scale = max(0.5, step_scale * 0.95)
        else:
            step_scale = min(2.0, step_scale * 1.05)
        
        if it % 100 == 99:
            h_ref = _peak_tracking_descent(h, max_rounds=5)
            c_ref = _cost(h_ref)
            if c_ref < cur_cost:
                h = h_ref
                cur_cost = c_ref
                if c_ref < best_cost:
                    best_cost = c_ref
                    best_h = h.copy()
    
    return best_h


def _try_start(h0, t_end):
    """Full pipeline: peak-tracking descent -> anneal -> final descent."""
    h = _peak_tracking_descent(h0, max_rounds=30)
    if time.time() > t_end:
        return h
    h = _anneal(h, t_end, n_iter=1500)
    if time.time() > t_end:
        return h
    h = _peak_tracking_descent(h, max_rounds=30)
    return h


def _optimize():
    t_end = START + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    best = np.ones(N) / N
    best_score = _ratio(best)

    starts = []

    # Single spikes
    for s in range(N):
        v = np.zeros(N); v[s] = 1.0
        starts.append(v)

    # Small blocks at various positions
    for w in (2, 3, 4, 5, 6, 8, 10):
        for s in range(0, N - w + 1, max(1, (N - w) // 8)):
            v = np.zeros(N); v[s:s + w] = 1.0 / w
            starts.append(v)

    # Symmetric patterns
    for k in range(2, 10):
        v = np.zeros(N)
        for i in range(k):
            v[i] = 1.0 / (k + i)
            v[N - 1 - i] = 1.0 / (k + i)
        v = v / v.sum()
        starts.append(v)

    # Asymmetric decaying patterns
    for alpha in (0.2, 0.5, 1.0, 2.0):
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

    # Bimodal patterns
    for gap in range(5, 20, 3):
        for width in (3, 5, 8):
            v = np.zeros(N)
            center1 = gap // 2
            center2 = N - 1 - gap // 2
            for i in range(N):
                d1 = abs(i - center1)
                d2 = abs(i - center2)
                v[i] = np.exp(-d1 / width) + np.exp(-d2 / width)
            if v.sum() > 0:
                v = v / v.sum()
                starts.append(v)

    # Random sparse starts
    for _ in range(40):
        v = rng.random(N)
        thresh = rng.uniform(0.3, 0.7)
        v[v < thresh] = 0.0
        if v.sum() > 0:
            v = v / v.sum()
            starts.append(v)

    # Random dense starts
    for _ in range(20):
        v = rng.random(N) + 0.01
        v = v / v.sum()
        starts.append(v)

    # Additional structured starts: periodic patterns
    for period in (3, 4, 5, 7):
        v = np.zeros(N)
        for i in range(N):
            v[i] = 1.0 if (i % period == 0) else 0.1
        v = v / v.sum()
        starts.append(v)

    # Power-law decays from center
    for p in (1.5, 2.0, 2.5, 3.0):
        v = np.zeros(N)
        center = N // 2
        for i in range(N):
            d = abs(i - center) + 1
            v[i] = 1.0 / (d ** p)
        v = v / v.sum()
        starts.append(v)

    # Two-bump with varying separation
    for sep in range(5, N - 5, 3):
        for width in (2, 3, 4):
            v = np.zeros(N)
            c1 = sep // 2
            c2 = N - 1 - sep // 2
            for i in range(N):
                v[i] = np.exp(-((i - c1) ** 2) / (2 * width ** 2)) + \
                       np.exp(-((i - c2) ** 2) / (2 * width ** 2))
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

    n_starts = len(starts)
    
    for idx, v in enumerate(starts):
        if time.time() > t_end:
            break
        remaining = t_end - time.time()
        if idx < n_starts // 3:
            local_end = min(t_end, time.time() + remaining * 0.3)
        elif idx < 2 * n_starts // 3:
            local_end = min(t_end, time.time() + remaining * 0.15)
        else:
            local_end = min(t_end, time.time() + remaining * 0.05)
        
        h = _try_start(v, local_end)
        sc = _ratio(h)
        if sc > best_score:
            best_score = sc
            best = h.copy()

    if time.time() < t_end:
        remaining = t_end - time.time()
        if remaining > 5:
            best = _peak_tracking_descent(best, max_rounds=50)
            best = _anneal(best, t_end, n_iter=3000)
            best = _peak_tracking_descent(best, max_rounds=50)
            best_score = _ratio(best)

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
