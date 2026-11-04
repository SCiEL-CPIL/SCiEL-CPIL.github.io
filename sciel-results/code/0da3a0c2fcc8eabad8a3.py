"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
The continuous autoconvolution sampled on the grid is approximated by the
discrete convolution (h*h)(x_i) = dh * sum_j h_j h_k over j+k = i + OFFSET,
where OFFSET = (1/2)/dh = 10 (because x_j + x_k = x_i  <=>  j + k = i + 10).

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf (equivalently maximize 1 / (||h||_1 * ||h*h||_inf)).

We use a soft-max (log-sum-exp) smoothing of the supremum so the objective
becomes differentiable and L-BFGS-B can converge to better local optima.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (1/2)/DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 150.0
_SAFETY = 12.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _soft_logsup(h, beta):
    """Smooth log(sup autoconv) via log-sum-exp, plus a stable shift."""
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 1e-30:
        return -np.inf
    # log(sup) approx = acmax + (1/beta) log sum exp(beta*(ac - acmax))
    shifted = ac - acmax
    return acmax + math.log(float(np.sum(np.exp(beta * shifted)))) / beta


def _log_ratio(h, beta):
    """log R = log l2sq - log l1 - log(sup). Returns -inf if invalid."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l2sq <= 1e-30 or l1 <= 1e-30:
        return -np.inf
    ls = _soft_logsup(h, beta)
    if not math.isfinite(ls):
        return -np.inf
    return math.log(l2sq) - math.log(l1) - ls


def _neg_logR(x, beta):
    x = np.maximum(x, 0.0)
    v = _log_ratio(x, beta)
    if not math.isfinite(v):
        return 1e12
    return -v


def _neg_logR_grad(x, beta):
    x = np.maximum(x, 0.0)
    val = _neg_logR(x, beta)
    g = np.zeros_like(x)
    eps = 1e-6
    for i in range(N):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] = max(x[i] - eps, 0.0)
        vp = _neg_logR(xp, beta)
        vm = _neg_logR(xm, beta)
        denom = xp[i] - xm[i]
        if denom <= 0.0:
            denom = eps
        g[i] = (vp - vm) / denom
    return val, g


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0.0:
        return h
    return h / s


def _local_optimize(h0, time_budget):
    """Optimize the smoothed objective (fix ||h||_2=1 via scale-invariance),
    then polish on the raw ratio. Returns (best_h, best_ratio)."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def _over():
        return (time.time() - t0) > time_budget

    # --- Smoothed L-BFGS-B on -log R (soft max) ---
    beta = 60.0
    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            lambda x: _neg_logR(x, beta), h0,
            method='L-BFGS-B',
            jac=lambda x: _neg_logR_grad(x, beta),
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12,
                     'gtol': 1e-9, 'disp': False},
            callback=lambda x: (np.nan if _over() else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rx = _normalize(rx)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    # --- Nelder-Mead polish on the raw (exact) ratio ---
    try:
        for _ in range(2):
            if _over() or _time_left() <= 5.0:
                break
            res2 = minimize(
                lambda x: -_ratio(np.maximum(x, 0.0)), best_h,
                method='Nelder-Mead',
                options={'maxiter': 250, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _initializations():
    """Yield a diverse set of promising starting points."""
    inits = []
    rng = np.random.default_rng(12345)

    # Single spikes at several positions
    for pos in [0, N // 4, N // 2, 3 * N // 4, N - 1]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)

    # Two spikes (spread to reduce autoconv peak overlap)
    for a in range(0, N, 8):
        for b in range(a + 10, N, 8):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)

    # Three spikes
    for a in range(0, N, 16):
        for b in range(a + 14, N, 16):
            for c in range(b + 14, N, 16):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)

    # Random sparse profiles
    for _ in range(45):
        k = int(rng.integers(1, 13))
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)

    # Random soft/sparse dense profiles
    for _ in range(45):
        s = rng.random(N) ** 3.0
        inits.append(s)

    # Gaussian bumps at several centers/widths
    for center in range(0, N, 6):
        for width in [2.0, 3.5, 5.0]:
            x = np.arange(N)
            s = np.exp(-0.5 * ((x - center) / width) ** 2)
            inits.append(s)

    # Two well-separated Gaussian bumps (reduces central autoconv peak)
    for a in range(3, N - 3, 6):
        for b in range(a + 10, N - 3, 6):
            x = np.arange(N)
            s = np.exp(-0.5 * ((x - a) / 2.5) ** 2) + np.exp(-0.5 * ((x - b) / 2.5) ** 2)
            inits.append(s)

    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    inits = _initializations()
    # Pre-evaluate and sort by ratio (best first) so budget goes to the top.
    scored = []
    for s in inits:
        v = _ratio(s)
        scored.append((v, s))
        if v > best_v:
            best_v = v
            best_h = s.copy()
    scored.sort(key=lambda t: -t[0])

    # Optimize the top initializations within the time budget.
    budget_per = 6.0
    for v0, s in scored:
        if _time_left() <= budget_per + 2.0:
            break
        # Skip clearly weak starts once we have a strong incumbent.
        if v0 < best_v * 0.4 and best_v > 20.0:
            continue
        h_opt, v_opt = _local_optimize(s, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
        if _time_left() <= 20.0:
            break

    # Final high-precision polish on the global best if time remains.
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize(best_h, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    # Return as a plain sequence of nonnegative floats (step heights).
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
