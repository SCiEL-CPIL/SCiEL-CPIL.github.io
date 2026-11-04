"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

def _compute_c4(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf

    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)

    # Vectorized log computation
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)

    # For numerical stability, subtract max before exp
    log_P_max = np.max(log_P)
    if np.isinf(log_P_max):
        return np.inf

    P_shifted = np.exp(log_P - log_P_max)
    P2_shifted = P_shifted * P_shifted

    int_P = np.sum(w_quad * P_shifted) * np.exp(log_P_max)
    int_P2 = np.sum(w_quad * P2_shifted) * np.exp(2 * log_P_max)

    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf

    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4(roots)

def _log_objective(log_roots):
    """Objective in log-space for unconstrained optimization."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    return _compute_c4(roots)

def _gradient_c4(roots):
    """Compute gradient of C4 with respect to roots using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)

    # P(x) = prod (x - r_i)^2
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)

    log_P_max = np.max(log_P)
    if np.isinf(log_P_max):
        return None

    P = np.exp(log_P - log_P_max)
    P2 = P * P

    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)

    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return None

    c4 = int_P2 / (int_P * int_P)

    # Gradient: dC4/dr_i = 2 * sum(w * P * (x - r_i) * (-2)) / (int_P * int_P) + ...
    # More carefully:
    # dP/dr_i = -2 * P * (x - r_i) / (x - r_i) = -2 * P / (x - r_i) ... 
    # Actually P = prod_j (x - r_j)^2, so dP/dr_i = P * sum_j d/d r_i log((x-r_j)^2)
    # = P * 2 * (-(x-r_i)/(x-r_i)) ... wait
    # d/d r_i log((x-r_j)^2) = 0 if j != i, = 2 * (-1)/(x-r_i) if j = i
    # So dP/dr_i = P * (-2/(x - r_i))
    # dP2/dr_i = 2 * P * dP/dr_i = 2 * P * P * (-2/(x-r_i)) = -4 * P2 / (x - r_i)

    grad = np.zeros(len(roots))
    for i, r in enumerate(roots):
        denom = x_quad - r
        # Avoid division by zero
        mask = np.abs(denom) > 1e-10
        if not np.any(mask):
            grad[i] = 0.0
            continue
        dP = -2.0 * P[mask] / denom[mask]
        dP2 = 2.0 * P[mask] * dP[mask]
        dinP = np.sum(w_quad[mask] * dP[mask])
        dinP2 = np.sum(w_quad[mask] * dP2[mask])
        # dC4 = dinP2 / int_P^2 - 2 * int_P2 * dinP / int_P^3
        grad[i] = dinP2 / (int_P * int_P) - 2.0 * int_P2 * dinP / (int_P * int_P * int_P)

    return grad

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill

    def _time_left():
        return deadline - time.time()

    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)

    best_roots = x0.copy()
    best_score = _compute_c4(best_roots)

    rng = np.random.default_rng(42)

    # ============================================================
    # PHASE 1: Multi-start local optimization in log-space
    # ============================================================
    log_x0 = np.log(x0)

    # Generate diverse initial points in log-space
    initial_points_log = [log_x0.copy()]

    # Perturbed Laguerre roots in log-space
    for scale in [0.1, 0.2, 0.3, 0.5, 0.8]:
        perturbed_log = log_x0 + scale * rng.standard_normal(12)
        perturbed_log = np.sort(perturbed_log)
        initial_points_log.append(perturbed_log)

    # Geometric spacings in log-space (linear in log)
    for log_start in [-3.0, -2.0, -1.0, 0.0, 1.0]:
        for log_end in [0.0, 1.0, 2.0, 3.0, 4.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                initial_points_log.append(log_roots.copy())

    # Random log-uniform
    for _ in range(10):
        log_roots = np.sort(rng.uniform(-3.0, 4.0, size=12))
        initial_points_log.append(log_roots.copy())

    # Structured: with varying spacing
    for base in [0.5, 1.0, 1.5, 2.0]:
        for spread in [0.3, 0.5, 0.7, 1.0]:
            log_roots = base + spread * np.linspace(0, 3, 12)
            initial_points_log.append(log_roots.copy())

    # Run local optimization from each initial point
    for log_init in initial_points_log:
        if _time_left() < 60:
            break

        try:
            result = minimize(
                _log_objective,
                log_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )

            candidate_roots = np.exp(result.x)
            if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                candidate_roots = np.sort(candidate_roots)
                score = _compute_c4(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots.copy()
        except Exception:
            pass

    # ============================================================
    # PHASE 2: Basin hopping in log-space
    # ============================================================
    if _time_left() > 120:
        try:
            def _log_wrapper(params):
                roots = np.exp(params)
                if np.any(roots <= 0) or np.any(np.isnan(roots)):
                    return np.inf
                return _compute_c4(roots)

            # Start from best so far
            log_best = np.log(best_roots)

            bh_result = basinhopping(
                _log_wrapper,
                x0=log_best,
                niter=30,
                minimizer_kwargs={'options': {'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}},
                stepsize=0.5,
                seed=42
            )

            if bh_result.fun < best_score:
                candidate_roots = np.exp(bh_result.x)
                if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                    candidate_roots = np.sort(candidate_roots)
                    score = _compute_c4(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()
        except Exception:
            pass

    # ============================================================
    # PHASE 3: Differential evolution in log-space
    # ============================================================
    if _time_left() > 90:
        try:
            log_bounds = [(-4.0, 5.0)] * 12

            result_de = differential_evolution(
                _log_objective,
                bounds=log_bounds,
                maxiter=50,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=False,
                updating='immediate',
                workers=1
            )

            if result_de.fun < best_score:
                candidate_roots = np.exp(result_de.x)
                if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                    candidate_roots = np.sort(candidate_roots)
                    score = _compute_c4(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()

            # Polish the DE result
            if np.all(result_de.x > -10):
                result_polish = minimize(
                    _log_objective,
                    result_de.x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if not np.any(np.isnan(result_polish.x)):
                    candidate_roots = np.exp(result_polish.x)
                    if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                        candidate_roots = np.sort(candidate_roots)
                        score = _compute_c4(candidate_roots)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate_roots.copy()
        except Exception:
            pass

    # ============================================================
    # PHASE 4: Final high-precision polish
    # ============================================================
    if _time_left() > 20:
        try:
            log_best = np.log(best_roots)
            for ftol, gtol, maxit in [(1e-14, 1e-12, 300), (1e-15, 1e-13, 300), (1e-16, 1e-14, 200)]:
                if _time_left() < 10:
                    break
                result_final = minimize(
                    _log_objective,
                    log_best,
                    method='L-BFGS-B',
                    options={'maxiter': maxit, 'ftol': ftol, 'gtol': gtol}
                )
                if not np.any(np.isnan(result_final.x)):
                    candidate_roots = np.exp(result_final.x)
                    if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                        candidate_roots = np.sort(candidate_roots)
                        score = _compute_c4(candidate_roots)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate_roots.copy()
                            log_best = result_final.x.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
