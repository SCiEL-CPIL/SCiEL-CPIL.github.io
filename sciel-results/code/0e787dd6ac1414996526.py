"""Uncertainty inequality: tune Hermite (deg 0,4,8) coeffs to reduce the C4 upper bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 180.0
_MARGIN = 6.0


def _hermite_poly(x, n):
    """Physicists' Hermite polynomial H_n via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(c0, c4, c8):
    """Return (A(g), A(g_hat), L2(g)) for the Hermite-Gaussian combination."""
    n = 2048
    L = 12.0
    x = np.linspace(-L, L, n, endpoint=False)
    dx = 2.0 * L / n
    e = np.exp(-0.5 * x * x)
    h0 = _hermite_poly(x, 0)
    h4 = _hermite_poly(x, 4)
    h8 = _hermite_poly(x, 8)
    h12 = _hermite_poly(x, 12)
    c12 = -(c0 * h0[0] + c4 * h4[0] + c8 * h8[0]) / h12[0]
    p = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    g = p * e
    ag = float(np.sum(np.abs(g)) * dx)
    l2 = float(np.sqrt(np.sum(g * g) * dx))
    g_hat = np.fft.fftshift(np.fft.fft(g)) * dx
    agh = float(np.sum(np.abs(g_hat)) * dx)
    return ag, agh, l2


def _objective(z):
    c0, c4, c8 = z
    ag, agh, l2 = _compute_bound(c0, c4, c8)
    return ag * agh + l2


def construct():
    global _DEADLINE
    # Prior state of the art (Gonçalves et al. 2017).
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = seed.copy()
    best_val = _objective(seed)
    rng = np.random.default_rng(2024)
    start = time.time()
    it = 0
    while time.time() + _MARGIN < _DEADLINE:
        it += 1
        if it <= 12:
            z0 = seed + rng.normal(0.0, 0.05, size=3)
        else:
            z0 = best_x + rng.normal(0.0, 0.15, size=3)
        try:
            res = minimize(_objective, z0, method="L-BFGS-B",
                           options=dict(maxiter=250, ftol=1e-12, gtol=1e-9))
            if res.success and res.fun < best_val:
                best_val = float(res.fun)
                best_x = np.asarray(res.x, dtype=float)
        except Exception:
            pass
    return best_x
# EVOLVE-BLOCK-END
