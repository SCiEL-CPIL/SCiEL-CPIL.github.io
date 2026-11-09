"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, gammaln, genlaguerre
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
import time
from math import comb, factorial, log, exp, lgamma

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    
    For the Cohn-Goncalves Laguerre construction, the bound is:
    C4 = ||P||^2_{L2(e^{-x})} / (prod(r_i))^2
    where P(x) = prod_i (x - r_i)^2
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    # Compute the L2 norm squared of P(x) = prod (x - r_i)^2 under weight e^{-x}
    # P(x)^2 = prod (x - r_i)^4
    # We need integral_0^inf prod (x - r_i)^4 * e^{-x} dx
    
    # Use the fact that for roots of generalized Laguerre polynomial,
    # this integral has a closed form via the norm of the Laguerre polynomial
    
    # Alternative: compute numerically using Gauss-Laguerre quadrature
    # with enough points to handle the polynomial
    
    n_roots = len(roots)
    # P(x) = prod (x - r_i)^2, so P(x)^2 = prod (x - r_i)^4
    # degree of P(x)^2 is 4*12 = 48
    
    # Use Gauss-Laguerre quadrature with enough nodes
    # For a polynomial of degree d, we need at least (d+1)/2 nodes
    # degree 48 -> need at least 25 nodes, use 100 for safety
    n_quad = max(100, 2 * n_roots + 50)
    
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x)^2 = prod (x - r_i)^4 at quadrature points
    log_P2 = np.zeros(n_quad)
    for r in roots:
        log_P2 += 4.0 * np.log(np.abs(x_quad - r))
    
    # Handle sign: P(x)^2 is always non-negative, but we need the actual value
    # prod (x-r_i)^4 is always >= 0
    P2_vals = np.exp(log_P2)
    
    # Integrate P(x)^2 * e^{-x} dx using Gauss-Laguerre (which integrates f(x)*e^{-x})
    norm_sq = np.sum(P2_vals * w_quad)
    
    # Product of roots
    log_prod_roots = np.sum(np.log(roots))
    prod_roots = exp(log_prod_roots)
    
    # C4 bound = norm_sq / (prod_roots)^2
    c4_bound = norm_sq / (prod_roots * prod_roots)
    
    return c4_bound

def _compute_c4_bound_fast(roots):
    """
    Faster computation using the property that for roots of L_m^{(alpha)},
    the norm has a closed form.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    # For the specific case where roots are zeros of L_12^{(12)}(x),
    # we can use the known norm formula
    # ||L_n^{(alpha)}||^2 = Gamma(n+alpha+1) / n!
    # But P(x) = prod (x-r_i)^2 = [L_12^{(12)}(x)]^2 * (leading coeff)^-2
    # This is getting complex, let's use numerical integration but faster
    
    n_quad = 80
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized computation
    # P2(x) = prod_i (x - r_i)^4
    # log(P2) = 4 * sum_i log|x - r_i|
    
    # Compute in log space for stability
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff))
    log_P2 = 4.0 * np.sum(log_abs_diff, axis=1)
    
    # Clip to avoid overflow
    log_P2 = np.clip(log_P2, -700, 700)
    P2_vals = np.exp(log_P2)
    
    norm_sq = np.sum(P2_vals * w_quad)
    
    log_prod_roots = np.sum(np.log(roots))
    prod_roots_sq = exp(2.0 * log_prod_roots)
    
    c4_bound = norm_sq / prod_roots_sq
    
    return c4_bound

def _check_constraints(roots, tol=1e-6):
    """
    Check if the roots satisfy the Laguerre-type polynomial constraints.
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    
    # Check that roots are distinct
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-8):
        return False
    
    return True

def _get_laguerre_zeros(m, alpha):
    """
    Get zeros of the generalized Laguerre polynomial L_m^{(alpha)}(x).
    Uses the companion matrix approach for numerical stability.
    """
    # Build the monic polynomial whose roots are the zeros of L_m^{(alpha)}
    # L_m^{(alpha)}(x) = sum_{k=0}^m (-1)^k * C(m+alpha, m-k) * x^k / k!
    
    # Coefficients of L_m^{(alpha)}(x) in ascending powers of x
    coeffs_asc = np.zeros(m + 1)
    for k in range(m + 1):
        # C(m+alpha, m-k) = Gamma(m+alpha+1) / (Gamma(m-k+1) * Gamma(k+alpha+1))
        if m - k < 0:
            coeffs_asc[k] = 0.0
        else:
            # Use log gamma for numerical stability
            log_coeff = lgamma(m + alpha + 1) - lgamma(m - k + 1) - lgamma(k + alpha + 1)
            coeff = exp(log_coeff)
            coeffs_asc[k] = ((-1)**k) * coeff / factorial(k)
    
    # For np.roots, we need highest degree first
    coeffs_desc = coeffs_asc[::-1]
    
    # Normalize to monic
    coeffs_desc = coeffs_desc / coeffs_desc[0]
    
    # Use companion matrix
    if m == 1:
        return [-coeffs_desc[1] / coeffs_desc[0]]
    
    # Build companion matrix
    C = np.zeros((m, m))
    C[np.arange(m-1), np.arange(1, m)] = 1.0
    C[m-1, :] = -coeffs_desc[1:] / coeffs_desc[0]
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(C)
    
    # Filter for real positive roots
    real_roots = []
    for ev in eigvals:
        if abs(ev.imag) < 1e-10 and ev.real > 1e-10:
            real_roots.append(ev.real)
    
    real_roots.sort()
    return real_roots

def _optimize_roots(initial_roots, deadline):
    """
    Optimize the root locations to minimize the C4 bound.
    Uses log-space parameterization for positivity constraint.
    """
    t_start = time.time()
    
    # Parameterize in log space: params = log(roots)
    # Objective: minimize C4 = norm_sq(P) / (prod roots)^2
    # In log space: minimize log(norm_sq) - 2*sum(log(roots))
    
    def log_objective(params):
        if time.time() > deadline - 5:
            return 1e10
        roots = np.exp(params)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return 1e10
        if not _check_constraints(roots):
            return 1e10
        return log(_compute_c4_bound_fast(roots))
    
    initial_params = np.log(initial_roots)
    
    # Bounds: roots should be in reasonable range
    lb = np.full(12, -5.0)   # roots > e^-5 ~ 0.0067
    ub = np.full(12, 15.0)   # roots < e^15 ~ 3.3e6
    bounds = list(zip(lb, ub))
    
    best_params = initial_params.copy()
    best_val = log_objective(initial_params)
    
    # Try multiple optimization methods and starting points
    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() > deadline - 10:
            break
        
        try:
            if method == 'L-BFGS-B':
                result = minimize(
                    log_objective,
                    best_params,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
            elif method == 'Nelder-Mead':
                result = minimize(
                    log_objective,
                    best_params,
                    method=method,
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            else:  # Powell
                result = minimize(
                    log_objective,
                    best_params,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12}
                )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Final polish with tight tolerances
    if time.time() < deadline - 15:
        try:
            result = minimize(
                log_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    final_roots = np.exp(best_params)
    if not _check_constraints(final_roots):
        final_roots = np.exp(initial_params)
    
    return final_roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin
    
    m = 12
    
    # Strategy: Start with Gaussian quadrature nodes for weight x^{m} e^{-x}
    # These are the zeros of L_m^{(m)}(x) = L_12^{(12)}(x)
    # This should be near-optimal for the moment-matching problem
    
    best_roots = None
    best_score = float('inf')
    
    # Primary approach: zeros of L_12^{(12)}
    try:
        roots_primary = _get_laguerre_zeros(m, m)
        if len(roots_primary) == 12:
            roots_primary = np.array(roots_primary)
            if _check_constraints(roots_primary):
                score = _compute_c4_bound_fast(roots_primary)
                if score < best_score:
                    best_score = score
                    best_roots = roots_primary.copy()
    except Exception:
        pass
    
    # Also try other alpha values as starting points
    alpha_values = [m, m-1, m+1, m//2, 2*m, m//4]
    for alpha in alpha_values:
        if time.time() > deadline - 30:
            break
        try:
            roots_trial = _get_laguerre_zeros(m, alpha)
            if len(roots_trial) == 12:
                roots_trial = np.array(roots_trial)
                if _check_constraints(roots_trial):
                    score = _compute_c4_bound_fast(roots_trial)
                    if score < best_score:
                        best_score = score
                        best_roots = roots_trial.copy()
        except Exception:
            continue
    
    # If we have good roots, optimize them
    if best_roots is not None and time.time() < deadline - 60:
        try:
            optimized_roots = _optimize_roots(best_roots, deadline)
            score_opt = _compute_c4_bound_fast(optimized_roots)
            if score_opt < best_score:
                best_score = score_opt
                best_roots = optimized_roots
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_roots is None:
        # Use standard Laguerre zeros L_12^{(0)}
        try:
            roots_fallback = _get_laguerre_zeros(m, 0)
            if len(roots_fallback) == 12:
                best_roots = np.array(roots_fallback)
            else:
                best_roots = np.array([4.0 * (i + 1) for i in range(12)])
        except Exception:
            best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Final verification
    best_roots = np.sort(best_roots)
    if not _check_constraints(best_roots):
        # Last resort: use well-spaced positive roots
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    return list(best_roots)

# EVOLVE-BLOCK-END
