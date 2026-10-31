"""First autocorrelation inequality: low-rank step function + DE search."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 135.0  # safety margin under 150s kill


def _objective(coeffs):
    """Max of the normalized autoconvolution over the 600 step intervals.

    coeffs = [a, b, c, d, e, f] with
      h_i = a + b*x_i + c*x_i^2 + d*max(x_i,0) + e*max(-x_i,0) + f*x_i^2*|x_i|
    x_i are the left endpoints of the 600 equal intervals on [-1/4, 1/4].
    """
    a, b, c, d, e, f = coeffs
    x = np.linspace(-0.25, 0.25, N + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    h = a + b * xm + c * xm * xm + d * np.maximum(xm, 0.0) + e * np.maximum(-xm, 0.0) + f * xm * xm * np.abs(xm)
    h = np.maximum(h, 0.0)  # nonnegative
    n = np.sum(h)
    if n <= 0:
        return 1e12
    conv = np.convolve(h, h, mode='full')
    return float(np.max(conv) / (n * n))


def _bounds():
    lo = [-0.5, -10.0, -10.0, -10.0, -10.0, -10.0]
    hi = [1.5, 10.0, 10.0, 10.0, 10.0, 10.0]
    return list(zip(lo, hi))


def construct():
    t0 = time.time()
    x = np.linspace(-0.25, 0.25, N + 1)
    xm = 0.5 * (x[:-1] + x[1:])

    # Strong seed: flat-top on |x| <= 1/8 (theoretical minimizer family).
    # Represented as a + c*x^2 with a=1, c=-16 so that h=1 for |x|<=1/8 and
    # h=0 at |x|=1/4, clamped to nonnegative.
    seed = np.array([1.0, 0.0, -16.0, 0.0, 0.0, 0.0])
    best_coeffs = seed.copy()
    best_val = _objective(seed)

    bounds = _bounds()

    # Multi-start differential evolution; stop when the deadline approaches.
    def time_check():
        return (time.time() - t0) > (DEADLINE - 12.0)

    pop = max(15, min(40, int((DEADLINE - 12.0) * 6)))
    maxiter = 60

    for restart in range(12):
        if time.time() - t0 > (DEADLINE - 12.0):
            break
        # Perturb the seed for diversity on later restarts.
        start = best_coeffs + (0.0 if restart == 0 else 0.3 * (np.random.rand(6) - 0.5))
        try:
            res = differential_evolution(
                _objective, bounds,
                x0=start,
                popsize=pop, maxiter=maxiter,
                tol=1e-8, mutation=(0.4, 1.0), recombination=0.8,
                seed=100 + restart, polish=False,
                callback=lambda *a, **k: None,
                init='latinhypercube',
            )
        except Exception:
            break
        if res.x is not None and res.fun < best_val:
            best_val = float(res.fun)
            best_coeffs = res.x.copy()
        if time.time() - t0 > (DEADLINE - 12.0):
            break

    # Final high-precision polish on the best coefficients.
    if time.time() - t0 < (DEADLINE - 12.0):
        try:
            res = differential_evolution(
                _objective, bounds,
                x0=best_coeffs,
                popsize=pop, maxiter=maxiter,
                tol=1e-12, mutation=(0.2, 0.8), recombination=0.9,
                seed=999, polish=True, polish_options={'maxiter': 2000, 'ftol': 1e-14},
            )
            if res.x is not None and res.fun < best_val:
                best_val = float(res.fun)
                best_coeffs = res.x.copy()
        except Exception:
            pass

    a, b, c, d, e, f = best_coeffs
    h = a + b * xm + c * xm * xm + d * np.maximum(xm, 0.0) + e * np.maximum(-xm, 0.0) + f * xm * xm * np.abs(xm)
    h = np.maximum(h, 0.0)
    return [float(v) for v in h]
