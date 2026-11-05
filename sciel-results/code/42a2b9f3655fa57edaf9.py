"""Structural change: parameterize 12 double roots as eigenvalues of a symmetric tridiagonal
Jacobi matrix with tunable recurrence coefficients, then refine in log-space with BFGS."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre, eval_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials up to 80
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf
    c4 = I2 / (I1 * I1)
    return float(c4)

def _c4_from_jacobi(diag, offdiag):
    """Compute C4 from eigenvalues of a symmetric tridiagonal Jacobi matrix.
    diag: length 12, offdiag: length 11 (all positive for positive eigenvalues)."""
    try:
        n = len(diag)
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = diag[i]
        for i in range(n - 1):
            J[i, i + 1] = offdiag[i]
            J[i + 1, i] = offdiag[i]
        eigvals = np.linalg.eigvalsh(J)
        eigvals = np.sort(eigvals)
        if np.any(eigvals <= 0):
            return np.inf
        return _compute_c4(eigvals)
    except Exception:
        return np.inf

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def _log_objective(log_roots):
    """Objective in log-space: C4 computed from exp(log_roots)."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or not np.all(np.isfinite(roots)):
        return 1e10
    c4 = _compute_c4(roots)
    if not np.isfinite(c4):
        return 1e10
    return c4

def _log_objective_grad(log_roots):
    """Objective and gradient in log-space."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or not np.all(np.isfinite(roots)):
        return 1e10, np.zeros_like(log_roots)
    
    c4 = _compute_c4(roots)
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(log_roots)
    
    # Numerical gradient in log-space
    eps = 1e-7
    grad = np.zeros_like(log_roots)
    for i in range(len(log_roots)):
        log_roots_plus = log_roots.copy()
        log_roots_plus[i] += eps
        f_plus = _log_objective(log_roots_plus)
        log_roots_minus = log_roots.copy()
        log_roots_minus[i] -= eps
        f_minus = _log_objective(log_roots_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return c4, grad

def construct():
    start = time.time()
    DEADLINE = 600.0
    SAFETY = 30.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY
    
    best_roots = None
    best_c4 = np.inf
    
    def update_best(roots):
        nonlocal best_roots, best_c4
        if roots is None:
            return
        roots = np.sort(np.asarray(roots, dtype=np.float64))
        if len(roots) != 12 or np.any(roots <= 0) or not np.all(np.isfinite(roots)):
            return
        c4 = _compute_c4(roots)
        if np.isfinite(c4) and c4 < best_c4:
            best_c4 = c4
            best_roots = roots.copy()
    
    # Phase 1: Search over generalized Laguerre alpha parameter
    # The roots of L_n^{(alpha)} are positive and ordered, giving a natural 1-parameter family.
    # Optimize alpha to find the best root density profile.
    
    # Coarse grid search over log-alpha
    log_alphas = np.linspace(math.log(1e-4), math.log(1e5), 300)
    for la in log_alphas:
        if not time_ok():
            break
        alpha = np.exp(la)
        c4_val = _c4_from_alpha(alpha)
        if c4_val < best_c4:
            best_c4 = c4_val
            xg, _ = roots_genlaguerre(12, alpha)
            best_roots = np.sort(xg)
    
    # Refine alpha with scalar minimization
    if time_ok() and best_roots is not None:
        # Find the alpha that gave best_c4
        best_alpha = None
        for la in log_alphas:
            alpha = np.exp(la)
            c4_val = _c4_from_alpha(alpha)
            if abs(c4_val - best_c4) < 1e-12:
                best_alpha = alpha
                break
        if best_alpha is None:
            best_alpha = 1.0
        
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(max(best_alpha * 0.5, 1e-5)), math.log(min(best_alpha * 2.0, 1e6))),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 500}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4:
                alpha_opt = np.exp(res_alpha.x)
                xg, _ = roots_genlaguerre(12, alpha_opt)
                update_best(xg)
        except Exception:
            pass
    
    # Also try standard Laguerre (alpha=0)
    if time_ok():
        try:
            x0, _ = roots_laguerre(12)
            update_best(x0)
        except Exception:
            pass
    
    # Phase 2: Jacobi matrix parameterization - optimize recurrence coefficients
    # The Jacobi matrix J = tridiag(b_i, a_i, b_i) has positive eigenvalues iff
    # it is positive definite. We parameterize to ensure positivity.
    # Use the fact that for orthogonal polynomials, the recurrence coefficients
    # determine the roots.
    
    if time_ok() and best_roots is not None:
        # Initialize from best roots: compute the Jacobi matrix that has these eigenvalues
        # For a symmetric tridiagonal matrix with given eigenvalues, we can use the
        # Stieltjes procedure or simply start from a diagonal matrix and add off-diagonals.
        
        # Start with diagonal = best_roots, off-diagonal = small positive values
        n = 12
        diag_init = best_roots.copy()
        offdiag_init = np.ones(n - 1) * 0.1
        
        def _jacobi_objective(params):
            """params: [diag_0, ..., diag_11, offdiag_0, ..., offdiag_10]"""
            diag = params[:n]
            offdiag = params[n:]
            return _c4_from_jacobi(diag, offdiag)
        
        # Use differential evolution to explore the Jacobi parameter space
        bounds_jacobi = [(1e-4, None)] * n + [(1e-6, None)] * (n - 1)
        
        try:
            de_result = differential_evolution(
                _jacobi_objective,
                bounds=bounds_jacobi,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                update_interval=10,
                polish=True
            )
            if de_result.success and np.isfinite(de_result.fun):
                diag_opt = de_result.x[:n]
                offdiag_opt = de_result.x[n:]
                J = np.zeros((n, n))
                for i in range(n):
                    J[i, i] = diag_opt[i]
                for i in range(n - 1):
                    J[i, i + 1] = offdiag_opt[i]
                    J[i + 1, i] = offdiag_opt[i]
                eigvals = np.linalg.eigvalsh(J)
                eigvals = np.sort(eigvals)
                if np.all(eigvals > 0):
                    update_best(eigvals)
        except Exception:
            pass
    
    # Phase 3: Log-space BFGS refinement from best roots
    if time_ok() and best_roots is not None:
        log_roots_init = np.log(best_roots)
        
        try:
            res_log = minimize(
                _log_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            if np.isfinite(res_log.fun):
                roots_candidate = np.exp(res_log.x)
                roots_candidate = np.sort(roots_candidate)
                if np.all(roots_candidate > 0):
                    update_best(roots_candidate)
        except Exception:
            pass
    
    # Phase 4: Multi-start in log-space with perturbed initializations
    if time_ok() and best_roots is not None:
        rng = np.random.default_rng(123)
        n_trials = 20
        for trial in range(n_trials):
            if not time_ok():
                break
            scale = 0.05 + 0.05 * trial
            pert = np.log(best_roots) + scale * rng.standard_normal(12)
            try:
                res = minimize(
                    _log_objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.isfinite(res.fun):
                    roots_candidate = np.exp(res.x)
                    roots_candidate = np.sort(roots_candidate)
                    if np.all(roots_candidate > 0):
                        update_best(roots_candidate)
            except Exception:
                pass
    
    # Phase 5: Try different alpha neighborhoods and refine
    if time_ok() and best_roots is not None:
        # Estimate current alpha from root distribution
        # For L_n^{(alpha)}, the roots have mean ~ (n + alpha) and variance ~ n*(n+alpha+1)
        mean_r = np.mean(best_roots)
        var_r = np.var(best_roots)
        # Rough estimate: alpha ~ mean - n for large n
        alpha_est = max(mean_r - 12, 0.1)
        
        for offset in [0.5, 2.0, 0.2, 5.0, 0.1, 0.8, 0.3, 3.0, 0.05, 10.0]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = alpha_est * offset if sign > 0 else alpha_est / offset
                if a_try < 1e-4 or a_try > 1e5:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    # Refine in log-space
                    log_init = np.log(xg2)
                    res2 = minimize(
                        _log_objective_grad,
                        log_init,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                    )
                    if np.isfinite(res2.fun):
                        roots_candidate = np.exp(res2.x)
                        roots_candidate = np.sort(roots_candidate)
                        if np.all(roots_candidate > 0):
                            update_best(roots_candidate)
                except Exception:
                    pass
    
    # Phase 6: Final polish with tight tolerances
    if time_ok() and best_roots is not None:
        log_roots_init = np.log(best_roots)
        try:
            res_final = minimize(
                _log_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50}
            )
            if np.isfinite(res_final.fun):
                roots_candidate = np.exp(res_final.x)
                roots_candidate = np.sort(roots_candidate)
                if np.all(roots_candidate > 0):
                    update_best(roots_candidate)
        except Exception:
            pass
    
    # Ensure all roots are positive and sorted
    if best_roots is None or np.any(best_roots <= 0):
        # Fallback: equally spaced
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = np.sort(best_roots)
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
