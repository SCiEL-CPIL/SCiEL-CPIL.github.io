"""Search program: optimize Hermite coefficients to minimize the C4 upper bound."""
import time
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize


# Precompute Hermite values at 0 for even degrees up to 12
# H_{2j}(0) = (-1)^j (2j)! / j!
def _H_at_0(deg):
    j = deg // 2
    return ((-1) ** j) * factorial(deg, exact=True) / factorial(j, exact=True)

H0 = _H_at_0(0)
H4 = _H_at_0(4)
H8 = _H_at_0(8)
H12 = _H_at_0(12)
# H0=1, H4=12, H8=-420, H12=11880


def _coeffs_and_norms(a0, a4, a8):
    a12 = -(a0 * H0 + a4 * H4 + a8 * H8) / H12
    coef = np.array([a0, a4, a8, a12])
    degs = np.array([0, 4, 8, 12])
    # ||g||^2 = sum a_k^2 k!   (probabilists' Hermite orthogonality)
    norms = coef * np.sqrt(factorial(degs, exact=True))
    return a12, norms


def _bound_value(x):
    a0, a4, a8 = x
    a12, norms = _coeffs_and_norms(a0, a4, a8)
    N = np.sum(norms ** 2)          # ||g||^2
    if N <= 1e-14:
        return 1e12
    # Fourier transform: F{H_k e^{-x^2/2}} = i^k H_k e^{-x^2/2}
    # |i^k|^2 = 1  =>  ||ĝ||^2 = ||g||^2 = N
    # A(f) * A(ĝ) <= sqrt( sum a_k^2 k! * sum a_k^2 (2k)! / k! )  = sqrt( sum a_k^2 (2k)! )
    # (Cauchy-Schwarz, termwise pairing of k-th coefficients)
    degs = np.array([0, 4, 8, 12])
    coef = np.array([a0, a4, a8, a12])
    cs = np.sqrt(np.sum((coef ** 2) * factorial(2 * degs, exact=True)))
    return cs / N


# ---------------------------------------------------------------------------
# Time-budgeted global + local search
# ---------------------------------------------------------------------------
def construct():
    deadline = time.time() + 195.0   # safety margin well under 210 s

    # Reference seed value (also a safe fallback)
    seed_pt = np.array([-113 / 100, 1 / 25, 1 / 3240], dtype=float)
    seed_val = _bound_value(seed_pt)
    best = seed_pt.copy()
    best_val = seed_val

    def update(x, v):
        nonlocal best, best_val
        if np.all(np.isfinite(x)) and np.isfinite(v) and v < best_val:
            best_val = v
            best = np.array(x, dtype=float)

    # ---- Stage 1: fine grid search around seed ----
    # The seed is known to be good; do a dense local grid to find improvements
    base = seed_pt.copy()
    scales = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    for scale in scales:
        if time.time() > deadline - 30.0:
            break
        # 7x7x7 grid around current best
        offsets = np.array([-1, -0.5, 0, 0.5, 1])
        for i in offsets:
            for j in offsets:
                for k in offsets:
                    if time.time() > deadline - 30.0:
                        break
                    pt = best + scale * np.array([i, j, k])
                    v = _bound_value(pt)
                    update(pt, v)
        # Polish the best found in this grid
        try:
            res = minimize(_bound_value, best, method='Nelder-Mead',
                           options={'xatol': 1e-14, 'fatol': 1e-15,
                                    'maxiter': 5000, 'adaptive': True})
            update(res.x, res.fun)
        except Exception:
            pass

    # ---- Stage 2: a few broader exploratory starts ----
    rng = np.random.default_rng(123)
    for _ in range(5):
        if time.time() > deadline - 15.0:
            break
        # Start from seed with moderate perturbation
        pt = seed_pt + 0.3 * rng.standard_normal(3)
        try:
            res = minimize(_bound_value, pt, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14,
                                    'maxiter': 3000, 'adaptive': True})
            update(res.x, res.fun)
        except Exception:
            pass

    # ---- Stage 3: final ultra-precise polish ----
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_bound_value, best, method='Nelder-Mead',
                           options={'xatol': 1e-15, 'fatol': 1e-16,
                                    'maxiter': 10000, 'adaptive': True})
            update(res.x, res.fun)
        except Exception:
            pass

    if not np.all(np.isfinite(best)):
        best = seed_pt.copy()

    return best
