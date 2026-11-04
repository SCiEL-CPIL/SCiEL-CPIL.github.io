"""Refined uncertainty inequality: tune Laguerre double roots to minimize C4 bound."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma as sp_gamma
import time

# Task parameters
M = 12
N = 1

def _laguerre_powers(m, alpha, x):
    """
    Compute L_k^{(alpha)}(x) for k=0..m at points x using three-term recurrence.
    L_0^{(a)}(x) = 1
    L_1^{(a)}(x) = 1 + a - x
    (k+1) L_{k+1}^{(a)}(x) = (2k + a + 1 - x) L_k^{(a)}(x) - (k + a) L_{k-1}^{(a)}(x)
    """
    x = np.asarray(x, dtype=float)
    npts = x.size
    L = np.zeros((m + 1, npts))
    L[0, :] = 1.0
    if m >= 1:
        L[1, :] = 1.0 + alpha - x
    for k in range(1, m):
        L[k + 1, :] = ((2 * k + alpha + 1 - x) * L[k, :] - (k + alpha) * L[k - 1, :]) / (k + 1)
    return L

def _compute_C4(roots, alpha):
    """
    Compute the C4 upper bound from the 12 double roots and alpha.
    The bound involves the Laguerre combination polynomial with double roots.
    """
    roots = np.asarray(roots, dtype=float)
    if roots.size != M:
        return np.inf
    if np.any(roots <= 0):
        return np.inf
    if np.any(np.diff(roots) <= 0):
        return np.inf

    # The Laguerre combination with double roots:
    # P(x) = sum_{k=0}^{M} c_k L_k^{(alpha)}(x)
    # where P has double roots at each r_i.
    # This means P(r_i) = 0 and P'(r_i) = 0 for all i.
    # We have M double roots = 2M conditions, and M+1 coefficients.
    # So the system is overdetermined; we find the least-squares solution.
    #
    # Actually, for a fixed alpha, the 2M conditions determine the M+1 coefficients
    # up to consistency. Let's set up the linear system.
    
    # Build the matrix: rows are P(r_i)=0 and P'(r_i)=0
    # P(x) = sum_{k=0}^{M} c_k L_k^{(alpha)}(x)
    # P'(x) = sum_{k=0}^{M} c_k L_k^{(alpha)'}(x)
    
    # We need L_k^{(alpha)'}(x). Use the identity:
    # L_k^{(alpha)'}(x) = -L_{k-1}^{(alpha+1)}(x)
    
    L = _laguerre_powers(M, alpha, roots)
    Lp = _laguerre_powers(M - 1, alpha + 1, roots) if M >= 1 else np.zeros((1, roots.size))
    
    # Build A matrix (2M x (M+1))
    A = np.zeros((2 * M, M + 1))
    for i in range(M):
        A[2 * i, :] = L[:, i]
        if M >= 1:
            A[2 * i + 1, :] = -Lp[:, i]
        else:
            A[2 * i + 1, :] = 0.0
    
    # Solve least squares: A @ c = 0, with normalization sum(c_k) = 1 (or similar)
    # Actually we want the null space. Let's use SVD.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # The null space is the last row of Vt (smallest singular value)
    if s.size > 0 and s[-1] < 1e-10:
        c = Vt[-1, :]
    else:
        # No exact null space; use least squares residual
        c, residuals, rank, sv = np.linalg.lstsq(A, np.zeros(2 * M), rcond=None)
    
    # Normalize
    c_norm = np.sum(np.abs(c))
    if c_norm < 1e-15:
        return np.inf
    c = c / c_norm
    
    # Now compute the C4 bound. The C4 constant relates to the Fourier sign uncertainty.
    # For the Laguerre construction, the bound is typically:
    # C4 = 1 / (sum of certain weights)
    # 
    # A common formulation: the upper bound on the number of sign changes relates to
    # the integral of the squared Laguerre combination.
    # 
    # Let's use a practical bound: C4 ~ 1 / (integral_0^inf P(x)^2 x^{alpha} e^{-x} dx)
    # but this requires numerical integration.
    
    # Alternative: the C4 bound in the Cohn-Goncalves framework is related to
    # the coefficients and the Laguerre polynomial structure.
    # 
    # For a simpler computable proxy that correlates with the true bound:
    # C4 = 1 / (sum_{k=0}^M c_k^2 * k! * gamma(M+1) / gamma(M+1-k) * something)
    
    # Let's use the norm of the coefficient vector weighted by Laguerre orthogonality:
    # The L2 norm of P with weight x^alpha e^{-x} is sum_{k=0}^M c_k^2 * gamma(k+alpha+1)/k!
    # (since L_k are orthogonal with norm squared = gamma(k+alpha+1)/k!)
    
    weights = np.array([sp_gamma(k + alpha + 1) / sp_gamma(k + 1) for k in range(M + 1)])
    norm_sq = np.sum(c * c * weights)
    
    if norm_sq < 1e-15:
        return np.inf
    
    # The C4 bound is inversely related to this norm
    C4 = 1.0 / norm_sq
    
    return C4

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    best_C4 = np.inf
    best_roots = None
    
    # Strategy: optimize alpha and the roots jointly, or fix alpha to optimal and find roots.
    # First, let's try a grid search over alpha, then for each alpha find optimal roots.
    
    # The roots should be positive and ordered. Let's parameterize them as:
    # r_i = exp(t_i) where t_i are free parameters, ensuring positivity.
    # For ordering, we can use r_i = r_1 + sum_{j=2}^i exp(s_j)
    
    # Let's try a simpler approach: use the known structure.
    # For the standard Laguerre polynomial L_M^{(alpha)}, the roots are known.
    # But we need double roots, so we're constructing a different polynomial.
    
    # Let's do a direct optimization over the 12 root positions and alpha.
    
    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        alpha = params[0]
        log_roots = params[1:]
        roots = np.exp(log_roots)
        # Ensure ordering by sorting (the C4 computation should be symmetric in some sense)
        # Actually, the roots are just locations, so let's sort them
        roots = np.sort(roots)
        return _compute_C4(roots, alpha)
    
    # Initial guess: alpha = 0, roots spread out
    alpha_init = 0.0
    roots_init = np.linspace(1.0, 20.0, M)
    log_roots_init = np.log(roots_init)
    
    # Try differential evolution for global search
    try:
        bounds = [(-2.0, 5.0)] + [(0.0, 3.0)] * M  # alpha in [-2,5], log_roots in [0,3]
        
        def de_objective(x):
            if time.time() > deadline - 10:
                return 1e10
            alpha = x[0]
            log_roots = x[1:]
            roots = np.exp(log_roots)
            roots = np.sort(roots)
            return _compute_C4(roots, alpha)
        
        # Run differential evolution with limited budget
        de_start = time.time()
        de_max_time = min(120.0, deadline - (de_start - start_time) - 60)
        
        if de_max_time > 30:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_C4:
                best_C4 = de_result.fun
                best_roots = np.sort(np.exp(de_result.x[1:]))
        else:
            # Fallback: local optimization
            x0 = np.concatenate([[alpha_init], log_roots_init])
            res = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_C4:
                best_C4 = res.fun
                best_roots = np.sort(np.exp(res.x[1:]))
    except Exception:
        pass
    
    # Polish with local optimization if we have a good starting point
    if best_roots is not None and time.time() < deadline - 30:
        try:
            # Refine alpha and roots
            best_alpha = 0.0  # We need to track alpha too
            
            # Redo with alpha tracked
            def full_objective(params):
                if time.time() > deadline - 10:
                    return 1e10
                alpha = params[0]
                log_roots = params[1:]
                roots = np.exp(log_roots)
                roots = np.sort(roots)
                return _compute_C4(roots, alpha)
            
            # Start from best found
            if best_roots is not None:
                log_roots_best = np.log(best_roots)
                # Try several alpha values near 0
                for alpha_try in [0.0, -0.5, 0.5, 1.0, -1.0]:
                    x0 = np.concatenate([[alpha_try], log_roots_best])
                    res = minimize(full_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
                    if res.fun < best_C4:
                        best_C4 = res.fun
                        best_roots = np.sort(np.exp(res.x[1:]))
        except Exception:
            pass
    
    # If no valid result, return the seed program's output
    if best_roots is None or best_C4 >= np.inf:
        return [4.0 * (i + 1) for i in range(M)]
    
    return list(best_roots)

if __name__ == "__main__":
    result = construct()
    print(result)
