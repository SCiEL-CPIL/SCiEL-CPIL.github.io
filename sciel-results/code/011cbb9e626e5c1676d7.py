import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf

    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)

    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)

    P = np.exp(log_P)
    P2 = P * P

    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)

    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf

    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _objective_log(log_roots):
    """Objective in log-space to keep roots positive."""
    roots = np.exp(log_roots)
    if np.any(roots > 1e6) or np.any(roots < 1e-8):
        return 1e10
    return _compute_c4_bound_fast(roots)

def _power_sums_to_roots(p):
    """Convert first 12 power sums to roots via Newton's identities + polynomial root finding."""
    # Newton's identities: for monic polynomial with roots r_i,
    # e_k = (1/k) * sum_{j=1}^{k} (-1)^{j-1} e_{k-j} p_j
    # where e_0 = 1
    n = 12
    e = np.zeros(n + 1)
    e[0] = 1.0
    for k in range(1, n + 1):
        s = 0.0
        for j in range(1, k + 1):
            s += (-1) ** (j - 1) * e[k - j] * p[j - 1]
        e[k] = s / k
    # Polynomial: x^n - e_1 x^{n-1} + e_2 x^{n-2} - ... + (-1)^n e_n
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1.0
    for k in range(1, n + 1):
        coeffs[k] = (-1) ** k * e[k]
    roots = np.roots(coeffs)
    # Keep only real positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    real_roots = real_roots[real_roots > 1e-10]
    return real_roots

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill

    def _elapsed_ok(margin=30.0):
        return (time.time() - start_time) < (deadline - margin)

    # Get standard Laguerre roots of degree 12 as a natural starting point
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)

    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)

    if not _elapsed_ok(120.0):
        return [float(xi) for xi in best_roots]

    rng = np.random.default_rng(42)

    # Generate diverse initial points
    initial_points = [x0.copy()]

    # Perturbed Laguerre roots
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)

    # Scaled Laguerre roots
    for scale in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0]:
        initial_points.append((x0 * scale).copy())

    # Random uniform
    for _ in range(5):
        rand_roots = rng.uniform(0.1, 50, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)

    # Geometric spacing variations
    for start in [0.3, 0.5, 1.0, 2.0]:
        for ratio in [1.2, 1.3, 1.5, 1.7, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 200 and geom[0] > 0.05:
                initial_points.append(geom.copy())

    # Power-sum-based starts: optimize power sums then convert
    # Start from Laguerre roots' power sums
    p_lag = np.array([np.sum(x0 ** k) for k in range(1, 13)])

    # Try optimizing in power-sum space (6 free params, rest derived)
    # For simplicity, optimize all 12 power sums but constrain via positivity
    # Actually, let's optimize the first 6 power sums and derive the rest
    # This is complex, so instead use log-space optimization with symmetry awareness

    # Optimize in log-space (ensures positivity)
    bounds_log = [(-10.0, 10.0)] * 12

    # Try L-BFGS-B from each initial point (in log-space)
    for x_init in initial_points:
        if not _elapsed_ok(60.0):
            break

        log_init = np.log(np.maximum(x_init, 1e-10))

        try:
            result = minimize(
                _objective_log,
                log_init,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )

            candidate = np.exp(result.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Also try direct coordinate optimization from best so far
    if _elapsed_ok(60.0):
        bounds = [(1e-6, None)] * 12
        try:
            result = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Nelder-Mead polish (derivative-free, good for final refinement)
    if _elapsed_ok(60.0):
        try:
            result = minimize(
                _objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final high-precision polish with L-BFGS-B
    if _elapsed_ok(30.0):
        bounds = [(1e-8, None)] * 12
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
