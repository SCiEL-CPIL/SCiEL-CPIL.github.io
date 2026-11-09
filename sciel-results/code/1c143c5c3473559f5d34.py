"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize_scalar

def construct():
    # Refined Cohn-Goncalves construction: the 12 double roots are not simply the
    # zeros of a single Laguerre polynomial.  Instead we consider a family of
    # Laguerre-type polynomials and choose the parameter alpha that minimises the
    # last (rightmost) sign-change location, which corresponds to the tightest
    # upper bound on C4.

    def last_root(alpha):
        # Compute roots of L_12^(alpha)(x)
        try:
            r, _ = roots_genlaguerre(12, alpha)
            r = np.real_if_close(r, tol=1000)
            r = np.sort(np.real(r))
            # Filter to positive roots only
            r = r[r > 0]
            if len(r) == 0:
                return np.inf
            return float(r[-1])
        except Exception:
            return np.inf

    # Search over alpha in a reasonable range.  alpha > -1 is required for
    # orthogonality on (0, inf) with weight x^alpha e^{-x}.
    # We scan a wide range to find the alpha giving the smallest last root.
    best_alpha = 0.0
    best_val = last_root(0.0)

    # Coarse scan
    alphas = np.linspace(-0.99, 5.0, 200)
    for a in alphas:
        v = last_root(a)
        if v < best_val:
            best_val = v
            best_alpha = a

    # Fine scan around the best alpha
    if np.isfinite(best_val):
        lo = best_alpha - 0.5
        hi = best_alpha + 0.5
        res = minimize_scalar(last_root, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-8, 'maxiter': 200})
        if res.success and res.fun < best_val:
            best_alpha = res.x
            best_val = res.fun

    # Also try a few specific "natural" values in case the scan missed them
    for a in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, -0.5, -0.1]:
        v = last_root(a)
        if v < best_val:
            best_val = v
            best_alpha = a

    # Final: compute the 12 roots for the best alpha
    roots, _ = roots_genlaguerre(12, best_alpha)
    roots = np.real_if_close(roots, tol=1000)
    roots = np.sort(np.real(roots))
    roots = [float(r) for r in roots if r > 0]

    # Ensure we have exactly 12 positive roots; if not, fall back to alpha=0
    if len(roots) != 12:
        roots, _ = roots_genlaguerre(12, 0)
        roots = sorted([float(r) for r in roots if r > 0])

    return roots
# EVOLVE-BLOCK-END
