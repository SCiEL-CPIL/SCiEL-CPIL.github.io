import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute transformation: monomial coeffs (ascending) -> Laguerre coeffs
    # L_k^(1)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, j + 1) / math.factorial(j)
    
    # To go from monomial (ascending) p to Laguerre c: c = A^{-T} p
    # i.e., p = A^T c, so c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(m + 1) + alpha  # k + alpha

    def roots_to_norm_sq(roots):
        """Compute squared Laguerre norm for monic poly with given roots."""
        # Build monic polynomial coefficients (ascending order)
        # np.poly returns descending: [1, e1, e2, ..., em]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending: [e_m, e_{m-1}, ..., e_1, 1]
        
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if not np.isfinite(norm_sq) or norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_from_roots(roots):
        """Objective: 1/norm_sq (minimize this)."""
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        ns = roots_to_norm_sq(roots)
        if ns <= 1e-10:
            return 1e10
        return 1.0 / ns

    def cost_from_log_roots(log_roots):
        roots = np.exp(log_roots)
        return cost_from_roots(roots)

    # Get natural starting points
    try:
        lag_roots = np.sort(roots_genlaguerre(m, alpha)[0])
        lag_roots = np.maximum(lag_roots, 1e-8)
        lag_log_roots = np.log(lag_roots)
    except Exception:
        lag_roots = 4.0 * (2.0 ** np.arange(12))
        lag_log_roots = np.log(lag_roots)

    # Strategy: Multi-start Nelder-Mead + DE in log-space
    log_low = -8.0
    log_high = 12.0
    bounds = [(log_low, log_high)] * 12

    best_val = 1e10
    best_log_roots = lag_log_roots.copy()
    best_roots = lag_roots.copy()

    # Update best
    v0 = cost_from_log_roots(best_log_roots)
    if v0 < best_val:
        best_val = v0

    # Generate diverse initial points
    rng = np.random.default_rng(12345)
    initial_points = [best_log_roots.copy()]

    # Scaled Laguerre roots
    for scale in [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0]:
        pt = np.log(np.maximum(lag_roots * scale, 1e-8))
        if np.all(pt > log_low) and np.all(pt < log_high):
            initial_points.append(pt)

    # Geometric sequences
    for base in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
        for start in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            pt = np.log(start) + np.arange(12) * np.log(base)
            if np.all(pt > log_low) and np.all(pt < log_high):
                initial_points.append(pt)

    # Random
    for _ in range(60):
        pt = rng.uniform(log_low, log_high, size=12)
        initial_points.append(pt)

    # Clustered around Laguerre zeros
    for _ in range(40):
        center = rng.uniform(0, 3)
        spread = rng.uniform(0.5, 4.0)
        pt = center + spread * rng.standard_normal(12)
        pt = np.clip(pt, log_low, log_high)
        initial_points.append(pt)

    # Sort by initial cost (best first)
    initial_points.sort(key=lambda x: cost_from_log_roots(x))

    # Run Nelder-Mead on each initial point
    for i, init_pt in enumerate(initial_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        try:
            result = minimize(
                cost_from_log_roots, init_pt,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Run differential evolution if time permits
    if time.time() - start_time < deadline - 60:
        for seed in [42, 123, 789]:
            if time.time() - start_time > deadline - 30:
                break
            try:
                de_result = differential_evolution(
                    cost_from_log_roots,
                    bounds,
                    seed=seed,
                    maxiter=100,
                    popsize=20,
                    tol=1e-10,
                    mutation=(0.3, 1.0),
                    recombination=0.8,
                    polish=True,
                    init='sobol'
                )
                if de_result.fun < best_val:
                    best_val = de_result.fun
                    best_log_roots = de_result.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Final high-precision polish with Nelder-Mead
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish = minimize(
                cost_from_log_roots, best_log_roots,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if polish.fun < best_val:
                best_val = polish.fun
                best_log_roots = polish.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Also try BFGS polish from best point
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            bfgs_result = minimize(
                cost_from_log_roots, best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 200}
            )
            if bfgs_result.fun < best_val:
                best_val = bfgs_result.fun
                best_log_roots = bfgs_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure positive roots
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-10)
        return list(best_roots)

    # Fallback
    try:
        lr = np.sort(roots_genlaguerre(m, alpha)[0])
        lr = np.maximum(lr, 1e-10)
        return list(lr)
    except Exception:
        return [4.0 * (2.0 ** i) for i in range(12)]
