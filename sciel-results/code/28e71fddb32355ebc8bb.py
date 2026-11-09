# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 590.0  # safety margin before 600s wall-clock kill


def _time_left():
    return _DEADLINE - time.time()


def _intervals(N, x):
    """Return 1-D interval family: for each i in [0,N), the set of c in [0,1]
    for which a *fixed* query point a is inside interval [x_i + c*i/N, +1/N].
    Represented as (lo, hi) in c-space (may be empty)."""
    # interval at c: [x_i + c*i/N, x_i + c*i/N + 1/N]
    # point a in interval  <=>  x_i + c*i/N <= a <= x_i + c*i/N + 1/N
    # For i == 0: interval is [x_0, x_0 + 1/N] for all c  -> covered iff a in [x_0, x_0+1/N]
    # For i > 0: c*i/N <= a - x_i <= c*i/N + 1/N
    #   =>  (a - x_i - 1/N) / (i/N) <= c <= (a - x_i) / (i/N)
    #   intersect with [0, 1]
    Nf = float(N)
    out = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        if i == 0:
            out[i, 0] = 0.0
            out[i, 1] = 1.0
        else:
            lo = (a - x[i] - 1.0 / Nf) * Nf / i
            hi = (a - x[i]) * Nf / i
            out[i, 0] = max(0.0, lo)
            out[i, 1] = min(1.0, hi)
    return out


def _union_length(N, x, n_grid=2048):
    """Exact (grid) estimate of the 1-D union length: measure of c in [0,1]
    for which there exists i with a query point a in the i-th interval.
    We fix a = 0 (by translation invariance of the union length w.r.t. a shift
    applied to all x_i, the union length is independent of a)."""
    global a
    a = 0.0
    c = np.linspace(0.0, 1.0, n_grid)
    # For each i, covered c-range where a=0 in [x_i + c*i/N, x_i + c*i/N + 1/N]
    covered = np.zeros(n_grid, dtype=bool)
    for i in range(N):
        if i == 0:
            # 0 in [x_0, x_0 + 1/N] for all c
            if 0.0 >= x[0] and 0.0 <= x[0] + 1.0 / N:
                covered[:] = True
        else:
            lo = (0.0 - x[i] - 1.0 / N) * N / i
            hi = (0.0 - x[i]) * N / i
            m = (c >= max(0.0, lo)) & (c <= min(1.0, hi))
            covered |= m
    # union length = measure of covered set (trapezoid over grid)
    if covered.all():
        return 1.0
    # count covered cells; use midpoint rule
    return float(np.sum(covered)) / (n_grid - 1) * 1.0


def _union_length_fast(N, x, n_grid=4096):
    """Vectorized union length over c grid for a=0."""
    c = np.linspace(0.0, 1.0, n_grid)
    covered = np.zeros(n_grid, dtype=bool)
    inv = 1.0 / N
    for i in range(N):
        if i == 0:
            if 0.0 >= x[0] and 0.0 <= x[0] + inv:
                covered[:] = True
        else:
            lo = (-x[i] - inv) * N / i
            hi = (-x[i]) * N / i
            lo = lo if lo > 0.0 else 0.0
            hi = hi if hi < 1.0 else 1.0
            if lo < hi:
                covered |= (c >= lo) & (c <= hi)
        if covered.all():
            break
    return float(np.sum(covered)) / (n_grid - 1)


def _optimize_1d(N, x0, budget_s):
    """Minimize 1-D union length over x in R^N. Returns best x and value."""
    best_x = np.array(x0, dtype=np.float64)
    best_v = _union_length_fast(N, best_x)

    def obj(v):
        return _union_length_fast(N, v)

    t_end = time.time() + budget_s
    # Several random restarts + Nelder-Mead
    rng = np.random.default_rng(7)
    n_restart = 0
    while time.time() < t_end and n_restart < 40:
        n_restart += 1
        x = np.array(best_x, dtype=np.float64)
        if n_restart > 1:
            x = rng.normal(0.0, 0.5, size=N)
        try:
            res = minimize(obj, x, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-7})
            if res.fun < best_v - 1e-9:
                best_v = float(res.fun)
                best_x = np.array(res.x, dtype=np.float64)
        except Exception:
            pass
        if time.time() > t_end - 5.0:
            break
    # Final coordinate polish
    for _ in range(3):
        if time.time() > t_end - 3.0:
            break
        for i in range(N):
            if time.time() > t_end - 3.0:
                break
            for scale in (0.1, 0.05, 0.01, 0.002):
                for d in (-1.0, 1.0):
                    trial = best_x.copy()
                    trial[i] += d * scale
                    v = _union_length_fast(N, trial)
                    if v < best_v - 1e-9:
                        best_v = v
                        best_x = trial
    return best_x, best_v


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Keich-style initial guess (known good seed for 1-D Kakeya)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Allocate budget: ~580s total usable. Split between x and y.
    total_budget = min(_time_left() - 15.0, 580.0)
    half = total_budget / 2.0

    x_opt, vx = _optimize_1d(N, keich, half)
    y_opt, vy = _optimize_1d(N, keich, _time_left() - 10.0)

    # Build 64x2 positions in (i outer, j inner) raster order
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    positions = np.stack([x_opt[i_flat], y_opt[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
