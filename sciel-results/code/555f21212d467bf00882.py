import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time


def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via three-term recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(1, n):
        h2 = 2.0 * x * h1 - 2.0 * k * h0
        h0, h1 = h1, h2
    return h1


def _objective(coeffs):
    """Compute ∫|g(x)|⁴ dx where g = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x²/2},
    with a12 fixed by g(0)=0."""
    a0, a4, a8 = coeffs
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 665280.0

    def integrand(x):
        h0 = _hermite(0, x)
        h4 = _hermite(4, x)
        h8 = _hermite(8, x)
        h12 = _hermite(12, x)
        poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
        return (poly ** 2) ** 2 * np.exp(-2.0 * x * x)

    val, _ = quad(integrand, -15.0, 15.0, limit=300, epsabs=1e-15, epsrel=1e-15)
    return val


def construct():
    t0 = time.time()
    deadline = t0 + 185.0  # safety margin before 210s kill

    # Seed (Goncalves et al. 2017)
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_coeffs)

    # Multiple starting points for global search
    starts = [
        best_coeffs.copy(),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.5, 0.02, -0.001]),
        np.array([-0.8, 0.06, 0.0005]),
        np.array([-2.0, 0.01, 0.002]),
        np.array([-1.13, 0.041, 0.00029]),
        np.array([-1.13, 0.039, 0.00031]),
        np.array([-1.2, 0.035, 0.0004]),
        np.array([-1.05, 0.045, 0.00025]),
    ]

    for sp in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective, sp, method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-16, "adaptive": True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs
