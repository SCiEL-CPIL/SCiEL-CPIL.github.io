"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) over the three free Hermite
coefficients (degrees 0, 4, 8). The degree-12 weight is fixed so g(0)=0.
The objective is the maximum of (1-g(lambda))/g(lambda) over lambda in a
bounded region where the Gaussian tail is still significant, which is a
finite, coefficient-dependent proxy for the sharp uncertainty bound.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0
_START = None

def _hermite_values(x):
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    x2 = x * x
    h4 = 16.0 * x2 * x2 - 48.0 * x2 + 12.0
    h8 = 256.0 * x2**4 - 3584.0 * x2**3 + 13440.0 * x2**2 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x2**6 - 212992.0 * x2**5 + 3317760.0 * x2**4
           - 25333440.0 * x2**3 + 100810080.0 * x2**2 - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12

def _g_fast(a, grid_data):
    grid, h0, h4, h8, h12, expf = grid_data
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * expf

def _bound_fast(a, grid_data):
    g = _g_fast(a, grid_data)
    # Objective: max of (1-g)/g where g>0; penalize g<=0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(g > 1e-15, (1.0 - g) / g, 1e12)
    ratio = np.where(np.isfinite(ratio), ratio, 1e12)
    return float(np.max(ratio))

def _bound_smooth_fast(a, grid_data, temperature=0.5):
    g = _g_fast(a, grid_data)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(g > 1e-15, (1.0 - g) / g, 1e12)
    ratio = np.where(np.isfinite(ratio), ratio, 1e12)
    ratio = np.clip(ratio, 1e-6, 1e12)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    return m + temperature * math.log(float(np.mean(s)))

def construct():
    global _START
    if _START is None:
        _START = time.time()

    # Grid: focus on [0, 12] where Gaussian is significant
    grid = np.linspace(0.0, 12.0, 60001)
    h0, h4, h8, h12 = _hermite_values(grid)
    expf = np.exp(-0.5 * grid * grid)
    grid_data = (grid, h0, h4, h8, h12, expf)

    best_a = _SEED.copy()
    best_val = _bound_fast(_SEED, grid_data)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize
    rng = np.random.default_rng(42)

    # Multi-start
    starts = [_SEED.copy()]
    for scale in (0.05, 0.1, 0.2, 0.3):
        for _ in range(4):
            starts.append(_SEED + scale * rng.standard_normal(3))
    for _ in range(15):
        starts.append(np.array([
            rng.uniform(-2.0, 0.5),
            rng.uniform(-0.2, 0.3),
            rng.uniform(-0.01, 0.02),
        ]))

    for s in starts:
        if remaining() < 5.0:
            break
        try:
            res = minimize(
                lambda x: _bound_smooth_fast(x, grid_data),
                s, method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_fast(cand, grid_data)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            continue

    # Polish on exact objective
    if remaining() > 4.0:
        try:
            res = minimize(
                lambda x: _bound_fast(x, grid_data),
                best_a, method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_fast(cand, grid_data)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Nelder-Mead polish
    if remaining() > 2.0:
        try:
            res = minimize(
                lambda x: _bound_fast(x, grid_data),
                best_a, method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-10, "fatol": 1e-10},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_fast(cand, grid_data)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
