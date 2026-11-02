"""Optimized program: second autocorrelation inequality.

Maximize  ||h||_2^2 / (||h||_1 * ||h*h||_inf)  over 50 nonnegative heights.
The ratio is scale-invariant, so we fix ||h||_1 = 1 and minimize
    ||h||_2^2 * ||h*h||_inf .
We use iterative reweighting with a FIXED peak index of the autoconvolution
per iteration (avoids the non-smooth max), diverse random starts, and a
coordinate polish.  A global best is cached so repeated construct() calls
return the best configuration found.
"""

import time
import numpy as np

N = 50
DELTA = 1.0 / N          # 0.02, interval of [-1/4, 1/4]
TIME_LIMIT = 150.0
SAFETY = 12.0            # stop this far before the hard limit
START = time.time()

# Precomputed autoconvolution (full) matrix: (h*h)[k] = c[k] . h,  k in [-(N-1), N-1]
# c[k, i] = h[i] * h[i - k]  (shifted index), built via the standard Toeplitz form.
def _build_conv_matrix():
    # full convolution length 2N-1
    L = 2 * N - 1
    C = np.zeros((L, N))
    for k in range(L):            # k = -(N-1) .. (N-1)
        shift = k - (N - 1)       # in [-(N-1), N-1]
        for i in range(N):
            j = i - shift
            if 0 <= j < N:
                C[k, i] = 1.0     # coefficient: h[i] * h[j]
    return C

C = _build_conv_matrix()

def _ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    conv = C @ h
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return 0.0
    return l2sq / (l1 * pinf)

def _cost(h):
    # scale-invariant cost to MINIMIZE: ||h||_2^2 * ||h*h||_inf with ||h||_1 = 1
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return np.inf
    hn = h / l1
    l2sq = float(hn @ hn)
    conv = C @ hn
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return np.inf
    return l2sq * pinf

def _reweight(h, peak, iters=200, eps=1e-15):
    """Multiplicative reweighting with fixed peak index `peak` of the conv.
    Minimizes ||h||_2^2 * conv[peak] subject to ||h||_1 = 1, h >= 0."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return h
    h = h / h.sum()
    for _ in range(iters):
        conv = C @ h
        target = conv[peak]
        if target <= eps:
            break
        # gradient of (l2sq * target) w.r.t. h_i  =  2 h_i * target + l2sq * (d target/d h_i)
        l2sq = float(h @ h)
        # d target / d h_i = C[peak, i]  (since target = C[peak].h)
        grad = 2.0 * h * target + l2sq * C[peak, :]
        # multiplicative (exponential) update in the feasible simplex direction
        g = grad - np.mean(grad)          # keep sum roughly fixed
        g = np.maximum(g, 0.0)
        step = 0.5
        h_new = h * np.exp(-step * g / (np.max(g) + eps))
        s = h_new.sum()
        if s <= eps:
            break
        h_new = h_new / s
        h = h_new
    return h

def _coord_polish(h, peak, rounds=60):
    """1-D line search per coordinate to minimize cost with fixed peak."""
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
            # try a few candidate values on [0, 3*mean]
            mean = h.sum() / N
            cands = [0.0, base, base * 0.5, base * 2.0, mean, 3.0 * mean, 6.0 * mean]
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

def _try_start(h0, peak, t_end):
    h = _reweight(h0, peak)
    if time.time() > t_end:
        return h
    # re-detect peak after reweighting, re-optimize once
    conv = C @ h
    peak2 = int(np.argmax(np.abs(conv)))
    h = _reweight(h, peak2, iters=100)
    h = _coord_polish(h, peak2)
    return h

def _optimize():
    t_end = START + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)
    best = np.ones(N) / N
    best_score = _ratio(best)

    # diverse starts
    starts = []
    # single spike
    for s in range(N):
        v = np.zeros(N); v[s] = 1.0; starts.append(v)
    # small blocks
    for w in (2, 3, 4):
        for s in range(0, N - w + 1, 2):
            v = np.zeros(N); v[s:s + w] = 1.0; starts.append(v)
    # random
    for _ in range(40):
        v = rng.random(N)
        v[v < 0.3] = 0.0
        if v.sum() > 0:
            v = v / v.sum()
            starts.append(v)

    # dedupe-ish by rounding
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
        # pick a peak: try the conv peak of the start and a couple of fixed peaks
        conv = C @ v
        peak = int(np.argmax(np.abs(conv)))
        h = _try_start(v, peak, t_end)
        sc = _ratio(h)
        if sc > best_score:
            best_score = sc
            best = h
        # also try a couple of alternative fixed peaks for diversity
        for alt in (peak, N - 1 - peak):
            if time.time() > t_end:
                break
            if alt == peak:
                continue
            h2 = _try_start(v, alt, t_end)
            sc2 = _ratio(h2)
            if sc2 > best_score:
                best_score = sc2
                best = h2

    return best, best_score

_cache = None

def construct():
    global _cache
    if _cache is None:
        _cache = _optimize()
    best, _ = _cache
    # return a plain python list of nonnegative floats
    return [float(x) for x in best]
