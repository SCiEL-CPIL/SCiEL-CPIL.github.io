import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute Chebyshev basis on [-1, 1]
_x = np.linspace(-1, 1, N)
_basis_cache = {}

def _get_basis(degree):
    if degree in _basis_cache:
        return _basis_cache[degree]
    B = np.zeros((degree + 1, N))
    for k in range(degree + 1):
        B[k] = eval_chebyt(k, _x)
    _basis_cache[degree] = B
    return B

def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _h_from_coeffs(coeffs, degree):
    B = _get_basis(degree)
    return coeffs @ B

def _objective_smooth(coeffs, beta=20.0):
    """Smooth objective using log-sum-exp for the max in the autocorrelation."""
    degree = len(coeffs) - 1
    h = _h_from_coeffs(coeffs, degree)
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    return abs(2.0 * N * soft_max / (s * s))

def _spectral_guided_init(best_h, degree):
    """Project best_h onto Chebyshev basis to get good initial coefficients."""
    B = _get_basis(degree)
    coeffs, _, _, _ = np.linalg.lstsq(B, best_h, rcond=None)
    return coeffs

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    def elapsed():
        return time.time() - t0

    def time_left():
        return DEADLINE - elapsed()

    # Phase 1: Global search via differential evolution in low-dim Chebyshev space
    # Try multiple degrees with more aggressive settings
    degrees_to_try = [4, 6, 8, 10, 12, 14]

    for degree in degrees_to_try:
        if time_left() < 25:
            break

        num_coeffs = degree + 1
        bounds = [(-5.0, 5.0)] * num_coeffs

        try:
            result = differential_evolution(
                lambda c: _objective_smooth(c, beta=12.0),
                bounds,
                seed=42 + degree,
                maxiter=80,
                popsize=20,
                tol=1e-7,
                init='sobol',
                polish=True,
                updating='deferred',
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if time_left() < 20:
                break

            h_candidate = _h_from_coeffs(result.x, degree)
            v = _value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()

        except Exception:
            pass

    # Phase 2: Local refinement from best solution using smooth objective
    if time_left() > 25:
        for degree in [6, 8, 10, 12, 14, 16]:
            if time_left() < 15:
                break
            try:
                coeffs_init = _spectral_guided_init(best_h, degree)

                res = minimize(
                    lambda c: _objective_smooth(c, beta=25.0),
                    coeffs_init,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                h_candidate = _h_from_coeffs(res.x, degree)
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass

    # Phase 3: Multi-start local refinement with diverse initializations
    if time_left() > 20:
        rng = np.random.default_rng(123)
        for trial in range(8):
            if time_left() < 12:
                break
            degree = 12
            try:
                coeffs_init = rng.normal(0, 1.5, degree + 1)

                res = minimize(
                    lambda c: _objective_smooth(c, beta=25.0),
                    coeffs_init,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                h_candidate = _h_from_coeffs(res.x, degree)
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass

    # Phase 4: Full-space gradient refinement from best low-dim solution
    if time_left() > 15:
        h0 = best_h.copy()

        try:
            res = minimize(
                _value,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 80, 'maxfun': 300, 'ftol': 1e-11}
            )
            if time_left() > 10:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Phase 5: Try structured patterns as additional candidates
    if time_left() > 12:
        x = np.arange(N)
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 59, 61, 67]:
            if time_left() < 5:
                break
            for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
                if time_left() < 3:
                    break
                h = 1.0 + amp * np.cos(2 * np.pi * k * x / N)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 6: Final local polish using smooth objective in low-dim space with higher beta
    if time_left() > 10:
        for degree in [10, 12, 14, 16, 18]:
            if time_left() < 5:
                break
            try:
                coeffs_init = _spectral_guided_init(best_h, degree)

                res = minimize(
                    lambda c: _objective_smooth(c, beta=50.0),
                    coeffs_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-15, 'gtol': 1e-11}
                )
                h_candidate = _h_from_coeffs(res.x, degree)
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass

    # Phase 7: Nelder-Mead polish in full space
    if time_left() > 8:
        h0 = best_h.copy()
        try:
            res = minimize(
                _value,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if time_left() > 3:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Phase 8: Final high-precision polish with very tight tolerances
    if time_left() > 5:
        degree = 14
        try:
            coeffs_init = _spectral_guided_init(best_h, degree)

            res = minimize(
                lambda c: _objective_smooth(c, beta=60.0),
                coeffs_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            h_candidate = _h_from_coeffs(res.x, degree)
            v = _value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()
        except Exception:
            pass

    # Phase 9: One more round of full-space refinement if time permits
    if time_left() > 4:
        h0 = best_h.copy()
        try:
            res = minimize(
                _value,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 400, 'ftol': 1e-14}
            )
            if time_left() > 2:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)

    return best_h
