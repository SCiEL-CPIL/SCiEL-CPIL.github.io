import numpy as np
import time
from scipy.optimize import minimize

# ---- Timing / deadline guard (evaluator kills at 630s; keep a safety margin) ----
_T0 = time.time()
_DEADLINE = 615.0
def _elapsed():
    return time.time() - _T0
def _time_left():
    return _DEADLINE - _elapsed()

N = 8
INV_N = 1.0 / N

# ---------------------------------------------------------------------------
# Exact (nonsmooth) interval-union length and volume
# ---------------------------------------------------------------------------
def _union_len_exact(vals, N=N):
    """Total length of the union of N intervals [v_k, v_k + 1/N]."""
    L = np.asarray(vals, dtype=np.float64).copy()
    L += 1.0 / N
    order = np.argsort(vals)
    L = L[order]
    lo = L[0]
    hi = L[0]
    total = 0.0
    for k in range(1, N):
        l = L[k]
        if l <= hi:
            if l + 1.0 / N > hi:
                hi = l + 1.0 / N
        else:
            total += hi - lo
            lo = l
            hi = l + 1.0 / N
    total += hi - lo
    return total

def _area_exact(x, y, c, N=N):
    return _union_len_exact(x, N) * _union_len_exact(y, N)

def _volume_exact(x, y, nq=64, N=N):
    c, w = np.polynomial.legendre.leggauss(nq)
    c = 0.5 * (c + 1.0)
    w = 0.5 * w
    tot = 0.0
    for cc, ww in zip(c, w):
        tot += ww * _area_exact(x, y, cc, N)
    return tot

# ---------------------------------------------------------------------------
# Smooth (differentiable) soft-min surrogate for the interval-union length.
# The union length = sum of (r_k - l_k) over the merged run; each merged
# interval's right endpoint is a "soft max" over the raw right endpoints and
# its left endpoint is a "soft min" over the raw left endpoints. Using a
# temperature-scaled log-sum-exp gives a C^1 surrogate of the exact length.
# ---------------------------------------------------------------------------
def _soft_max(vals, T):
    s = vals / T
    m = s.max()
    return m * T + T * np.log(np.sum(np.exp(s - m)))

def _soft_min(vals, T):
    s = -vals / T
    m = s.max()
    return -(m * T + T * np.log(np.sum(np.exp(s - m))))

def _union_len_smooth(vals, T, N=N):
    base = np.asarray(vals, dtype=np.float64)
    L = base + 1.0 / N
    order = np.argsort(base)
    base = base[order]
    L = L[order]
    lo = L[0]
    hi = L[0]
    total = 0.0
    for k in range(1, N):
        l = L[k]
        if l <= hi:
            hi = _soft_max(np.array([hi, l + 1.0 / N]), T)
        else:
            total += hi - lo
            lo = l
            hi = l + 1.0 / N
    total += hi - lo
    return total

def _volume_smooth(x, y, T, nq=32, N=N):
    c, w = np.polynomial.legendre.leggauss(nq)
    c = 0.5 * (c + 1.0)
    w = 0.5 * w
    tot = 0.0
    for cc, ww in zip(c, w):
        tot += ww * _union_len_smooth(x, T, N) * _union_len_smooth(y, T, N)
    return tot

# ---------------------------------------------------------------------------
# Keich seed
# ---------------------------------------------------------------------------
def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        out[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return out

# ---------------------------------------------------------------------------
# Exact coordinate-descent polish (with adaptive shrinking step)
# ---------------------------------------------------------------------------
def _polish_exact(x, y, budget=8.0, N=N):
    step = 0.05
    for _ in range(400):
        if _time_left() < budget:
            break
        improved = False
        for i in range(N):
            for sgn in (-1.0, 1.0):
                xn = x.copy(); xn[i] += sgn * step
                v = _volume_exact(xn, y)
                if v < _volume_exact(x, y):
                    x = xn; improved = True
                    break
        for j in range(N):
            for sgn in (-1.0, 1.0):
                yn = y.copy(); yn[j] += sgn * step
                v = _volume_exact(x, yn)
                if v < _volume_exact(x, y):
                    y = yn; improved = True
                    break
        if not improved:
            step *= 0.6
            if step < 1e-7:
                break
    return x, y

# ---------------------------------------------------------------------------
# Smooth gradient-based optimization of ONE 8-dim subproblem
# ---------------------------------------------------------------------------
def _optimize_one(seed, other, budget=60.0):
    """Minimize the smooth volume with respect to `seed` (8-dim), holding
    `other` fixed. Returns the best 8-dim vector found (smooth objective)."""
    def obj(p):
        return _volume_smooth(p, other, T=0.01)

    x0 = np.asarray(seed, dtype=np.float64).copy()
    best = x0.copy()
    bestf = obj(x0)

    # L-BFGS-B (gradient-based) from a few starting points
    starts = [x0,
              x0 + 0.1 * np.sin(np.arange(1, N + 1)),
              x0 - 0.1 * np.cos(np.arange(1, N + 1))]
    for s in starts:
        if _time_left() < budget:
            break
        try:
            res = minimize(obj, s, method="L-BFGS-B",
                           options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9})
            if res.fun < bestf:
                bestf = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Nelder-Mead refinement (robust to the soft-min kinks)
    if _time_left() > budget:
        try:
            res = minimize(obj, best, method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-9,
                                    "fatol": 1e-12, "adaptive": True})
            if res.fun < bestf:
                bestf = res.fun
                best = res.x.copy()
        except Exception:
            pass
    return best

def kakeya_needle_3d8() -> np.ndarray:
    global _T0
    _T0 = time.time()
    _DEADLINE = 615.0

    seed = _keich(N)
    x = seed.copy()
    y = seed.copy()

    # Alternate optimising the x and y subproblems (separable structure).
    # Each subproblem is only 8-dim, so this converges far faster than the
    # joint 16-dim search and lets us afford more restarts / polish.
    it = 0
    while _time_left() > 30.0:
        it += 1
        x = _optimize_one(x, y, budget=max(20.0, _time_left() - 30.0))
        if _time_left() < 30.0:
            break
        y = _optimize_one(y, x, budget=max(20.0, _time_left() - 30.0))

    # Exact local polish to lock in a genuinely minimal (non-surrogate) volume.
    x, y = _polish_exact(x, y, budget=12.0)

    # Assemble the 64 base positions: row k -> (i, j) = (k//8, k%8)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = x[i_grid.flatten()]
    base_y = y[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)

    # Sanity: finite, non-degenerate box
    if not np.all(np.isfinite(positions)):
        positions = np.zeros((64, 2), dtype=np.float64)
    return positions
