"""Optimize the Hermite-Gaussian uncertainty bound via direct numerical objective."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, brentq, basin_hopping

_DEADLINE = time.time() + 200.0          # safety margin vs 210 s wall-clock
_ALPHA = 0.5                              # Gaussian width (scale-invariant)
_BETA = 1.0                               # Hermite argument scaling
_C12 = -1.0 / 3.0                         # H12 coefficient fixing g(0)=0
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _hermite_combo(x, c0, c4, c8, c12):
    """Value of sum_k c_k H_k(x) (physicist's Hermite polynomials)."""
    return (c0 * eval_hermite(0, x) + c4 * eval_hermite(4, x)
            + c8 * eval_hermite(8, x) + c12 * eval_hermite(12, x))


def _g(c0, c4, c8, x):
    return np.exp(-_ALPHA * x * x) * _hermite_combo(_BETA * x, c0, c4, c8, _C12)


def _g_hat(c0, c4, c8, x):
    # Fourier transform of e^{-a x^2} H_n(b x) = (1/sqrt(pi a)) e^{-y^2/(4a)}
    #   * (-i b)^n H_n(y/(2b))
    y = x / (2.0 * _BETA)
    scale = 1.0 / np.sqrt(np.pi * _ALPHA)
    f0 = scale * np.exp(-y * y / (4.0 * _ALPHA))
    f4 = scale * (1.0) * np.exp(-y * y / (4.0 * _ALPHA)) * eval_hermite(4, y)
    f8 = scale * (-1.0) * np.exp(-y * y / (4.0 * _ALPHA)) * eval_hermite(8, y)
    f12 = scale * (1.0) * np.exp(-y * y / (4.0 * _ALPHA)) * eval_hermite(12, y)
    return (c0 * f0 + c4 * f4 + c8 * f8 + _C12 * f12)


def _sup2(func, xlo=-12.0, xhi=12.0, n_grid=400, n_starts=12,
          tol=1e-12, maxiter=80):
    """Minimum of |func(x)|^2 over [xlo, xhi] -> (min_value, x_at_min)."""
    xs = np.linspace(xlo, xhi, n_grid)
    vals = np.abs(func(xs)) ** 2
    best_val = vals.min()
    best_x = xs[int(vals.argmin())]
    order = np.argsort(vals)[:n_starts]
    for i in order:
        x0 = float(xs[i])
        def obj(x):
            return float(np.abs(func(np.array(x))) ** 2)
        res = minimize(obj, [x0], method='Brent',
                       options={'xtol': tol, 'maxiter': maxiter})
        if res.success:
            xv = float(res.fun)
            if xv < best_val - 1e-15:
                best_val = xv
                best_x = float(res.x[0])
    return best_val, best_x


def _objective(v):
    """A(g) * A(g_hat) for c0=1, c4=v[0], c8=v[1]."""
    c4, c8 = float(v[0]), float(v[1])
    v1, _ = _sup2(lambda x: _g(1.0, c4, c8, x))
    v2, _ = _sup2(lambda x: _g_hat(1.0, c4, c8, x))
    return np.sqrt(max(v1, 0.0)) * np.sqrt(max(v2, 0.0))


def construct():
    global _DEADLINE
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # Reference (seed) score, using c0=1 normalization.
    c4s, c8s = _SEED[1] / _SEED[0], _SEED[2] / _SEED[0]
    best_val = _objective([c4s, c8s])
    best_v = np.array([c4s, c8s])

    def step(x):
        if elapsed() > _DEADLINE - 15.0:
            return float('inf')
        v = np.array(x, dtype=float)
        v[0] = np.clip(v[0], -10.0, 10.0)
        v[1] = np.clip(v[1], -10.0, 10.0)
        val = _objective(v)
        if np.isfinite(val) and val < best_val:
            nonlocal best_val, best_v
            best_val = val
            best_v = v.copy()
        return val

    # Deterministic multi-start local minimization first.
    rng = np.random.default_rng(12345)
    for r in (0.0, 0.3, 1.0, 3.0, 5.0):
        if elapsed() > _DEADLINE - 20.0:
            break
        for ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            if elapsed() > _DEADLINE - 20.0:
                break
            v0 = np.array([r * np.cos(ang), r * np.sin(ang)])
            res = minimize(_objective, v0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-9,
                                    'maxiter': 200})
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_v = np.array(res.x, dtype=float)

    # Global basin-hopping to escape local minima (structural global search).
    if elapsed() < _DEADLINE - 25.0:
        bh = basin_hopping(step, best_v.copy(), niter=40,
                           stepsize=1.5, minimizer_kwargs={'method': 'Nelder-Mead',
                                                           'options': {'maxiter': 60}},
                           seed=99)
        if np.isfinite(bh.fun) and bh.fun < best_val:
            best_val = float(bh.fun)
            best_v = np.array(bh.x, dtype=float)

    # Convert back to the required three coefficients (c0, c4, c8).
    return np.array([1.0, best_v[0], best_v[1]])
# EVOLVE-BLOCK-END
