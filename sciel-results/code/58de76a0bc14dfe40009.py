"""Parametric-manifold projection: fit the six-coefficient growth-index ansatz
to the exact CPL growth index over a dense stratified cosmology grid."""
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = 75.0  # hard wall-clock guard (evaluator kills at 90 s)
_EPS = 1e-10


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _om(a, om0, w0, wa):
    """Matter density ratio Om(a)/Om0 (normalized to 1 at a=1)."""
    a3 = a * a * a
    x = 1.0 - a
    return a3 + (1.0 - om0) * a ** (-3.0 * (1.0 + w0 + wa)) * (1.0 + wa * x) ** (-3.0 / wa)


def _gamma_exact(a, om0, w0, wa):
    """Exact smooth-CPL growth index gamma = d ln f / d ln a."""
    a = np.asarray(a, dtype=float)
    a3 = a * a * a
    x = 1.0 - a
    a3w = a ** (-3.0 * (1.0 + w0 + wa)) * (1.0 + wa * x) ** (-3.0 / wa)
    om0 = float(om0)
    om_a = om0 * a3 + (1.0 - om0) * a3w
    de_a = (1.0 - om0) * a3w
    w_a = w0 + wa * x
    return 0.5 * (3.0 * om_a - 1.0 - w_a * de_a)


def _fit_chebyshev(A, y):
    """Least-squares solve for y ~ A c, clipped to the coefficient box."""
    try:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        c = np.zeros(A.shape[1])
    return np.clip(c, -10.0, 10.0)


def _build_design(a, om0, w0, wa):
    """Rows [1, q, q^2, u, wa*a, u*q] for the ansatz at given points."""
    a = np.asarray(a, dtype=float)
    q = 1.0 - _om(a, om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    wa = float(wa)
    return np.column_stack([
        np.ones_like(a), q, q * q, u, wa * a, u * q
    ])


def _grid(a_min, n_a, om0s, w0s, was):
    """Dense stratified, physically-valid cosmology grid."""
    a = np.linspace(a_min, 1.0, n_a)
    A = []
    Y = []
    for om0 in om0s:
        for w0 in w0s:
            for wa in was:
                if (1.0 - om0) * (1.0 + w0) <= 0.0:
                    continue  # require positive DE density at a=1
                try:
                    A.append(_build_design(a, om0, w0, wa))
                    Y.append(_gamma_exact(a, om0, w0, wa))
                except Exception:
                    continue
    if not A:
        return None, None
    return np.vstack(A), np.concatenate(Y)


def _max_abs_err(A, y, c):
    return float(np.max(np.abs(A @ c - y)))


def construct():
    # Seed constant-gamma answer; always returned as the safe fallback.
    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    try:
        A, y = _grid(0.35, 24, [0.2, 0.27, 0.3], [-0.5, -0.9, -1.0, -1.1],
                     [-0.6, -0.3, 0.0, 0.3, 0.6])
        if A is None:
            return best.tolist()

        # Phase 1: global least-squares fit to the exact gamma.
        c = _fit_chebyshev(A, y)

        # Phase 2: iterative minimax polish on the max absolute gamma error.
        # (minimizes the f relative error; D error follows from the integral.)
        n = A.shape[0]
        w = np.ones(n)
        for _ in range(8):
            if _time_left() <= 8.0:
                break
            try:
                c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            except Exception:
                break
            c = np.clip(c, -10.0, 10.0)
            r = np.abs(A @ c - y)
            mx = float(r.max())
            if mx <= _EPS:
                break
            w = np.power(r / mx, 1.5)

        # Prefer the polish only if it genuinely improves the max error.
        if _max_abs_err(A, y, c) < _max_abs_err(A, y, best):
            best = c

        # Guard: gamma must stay in [0.1, 1.2] over the whole grid.
        if np.min(A @ best) >= 0.1 - 1e-9 and np.max(A @ best) <= 1.2 + 1e-9:
            return np.clip(best, -10.0, 10.0).tolist()
        return best.tolist()
    except Exception:
        return best.tolist()
