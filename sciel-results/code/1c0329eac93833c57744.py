"""Refined Cohn-Goncalves Laguerre construction: fast basis-eval + multi-start DE + L-BFGS."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def construct():
    m = 12

    # Build A: A[k,j] = coeff of x^j in L_k^(1)(x) = (-1)^j * binom(k+1,j+1)/j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    # M[j,k] = A[k,j]; p = M @ c  =>  c = M^{-1} p
    M = A.T
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    weights = (np.arange(m + 1) + 1).astype(float)  # ||L_k||^2 = k+1 for alpha=1

    def _norm_sq(roots):
        """
        roots: (12,) array of positive values.
        Returns squared L2 norm of P in Laguerre basis.
        Uses direct basis evaluation: L_k(r) computed via recurrence.
        """
        # Build p coefficients (ascending) from roots
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        if len(p) < m + 1:
            p = np.pad(p, (0, m + 1 - len(p)))
        else:
            p = p[:m + 1]

        # c = M_inv @ p
        c = M_inv @ p
        return float(np.sum(c * c * weights))

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return 1e10
        ns = _norm_sq(roots)
        if ns <= 1e-10:
            return 1e10
        return 1.0 / ns

    # Bounds for log roots: allow wide range
    bounds = [(-4.0, 12.0)] * 12

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # ---- Phase 1: Multi-start Differential Evolution ----
    # Use several different seeds / initializations
    seeds = [42, 123, 7, 999, 2024]
    popsizes = [15, 20, 12]

    for si, seed in enumerate(seeds):
        if _time_left() < 60:
            break
        ps = popsizes[si % len(popsizes)]
        try:
            result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=150,
                popsize=ps,
                tol=1e-8,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # ---- Phase 2: Additional local refinement from multiple starting points ----
    if _time_left() > 30:
        # Generate diverse starting points for L-BFGS-B
        start_points = []
        if best_log_roots is not None:
            start_points.append(best_log_roots)
        # Geometric sequence seeds
        for a in [2.0, 4.0, 8.0, 1.0, 0.5]:
            for q in [1.5, 2.0, 2.5, 3.0]:
                lr = np.log(a) + np.arange(12) * np.log(q)
                if np.all(lr >= bounds[0][0]) and np.all(lr <= bounds[0][1]):
                    start_points.append(lr)
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            lr = rng.uniform(bounds[0][0], bounds[0][1], size=12)
            start_points.append(lr)

        for sp in start_points:
            if _time_left() < 20:
                break
            try:
                res = minimize(
                    cost_func,
                    x0=sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # ---- Phase 3: High-precision final polish ----
    if _time_left() > 15 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func,
                x0=best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure valid output
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    # Absolute fallback: geometric sequence
    return [4.0 * (2.0 ** i) for i in range(12)]
