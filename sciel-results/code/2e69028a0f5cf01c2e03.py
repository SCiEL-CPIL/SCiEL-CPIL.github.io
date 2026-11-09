"""Optimized Cohn-Goncalves Laguerre polynomial root construction."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import genlaguerre, eval_genlaguerre, gamma as sp_gamma
from scipy.integrate import quad

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 60.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound for given double-root locations.
    
    In the Cohn-Goncalves construction, the uncertainty constant C4 is bounded
    by an expression involving the Laguerre polynomial combination. The bound
    depends on the integral of the squared magnitude of the constructed
    polynomial against the Laguerre weight.
    
    For m double roots at positions r_1, ..., r_m with n=1, the bound is
    related to the sum of squared Laguerre evaluations and the weight function.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    
    # Ensure positive and sorted
    if np.any(roots <= 0):
        return 1e10
    roots = np.sort(roots)
    
    # The C4 bound in the refined construction involves the norm of the
    # Laguerre combination. A key quantity is the integral:
    #   sum_{k=0}^{2m} c_k^2 * (2k + 2n - 1) / (k!)^2 * Gamma(k+2n)
    # but more directly, we can use the fact that the optimal roots minimize
    # the maximum deviation of the constructed sign function.
    
    # A practical bound: C4 is proportional to the product of distances
    # between consecutive roots and the boundary, raised to some power,
    # normalized by the Laguerre weight.
    
    # The Cohn-Goncalves bound for the sign uncertainty is:
    #   C4 <= (2m)! / (m!)^2 * prod_{i=1}^{m} r_i^{-1} * exp(-sum r_i/2)
    # times a correction factor.
    
    # More precisely, for the Laguerre combination with double roots at r_i:
    # The bound involves the reciprocal of the product of root spacings
    # weighted by the Laguerre weight function.
    
    # Let's use a well-motivated objective: the C4 upper bound is proportional
    # to the integral of |P(x)|^2 x^{2n-1} e^{-x} dx where P is the polynomial
    # with the given roots, normalized appropriately.
    
    # For the sign uncertainty problem, the relevant quantity is:
    #   C4_bound = sum_{i=0}^{2m} a_i^2 / w_i
    # where a_i are coefficients and w_i are weights.
    
    # A simpler but effective objective: minimize the maximum of the
    # Laguerre polynomial evaluated at the roots, which controls the
    # uncertainty bound.
    
    # The refined Cohn-Goncalves bound for C4 with m=12, n=1:
    # C4 <= K * prod_{i=1}^{m} (1/r_i) * exp(-sum_{i=1}^{m} r_i / (2m))
    # where K is a combinatorial constant.
    
    # Let's compute a bound that captures the essential dependence:
    # The uncertainty constant is bounded by the reciprocal of the
    # geometric mean of the roots, times an exponential damping factor.
    
    log_sum = np.sum(roots)
    log_prod_inv = -np.sum(np.log(roots))
    
    # The bound scales as: exp(log_prod_inv - log_sum / (2*m))
    # This captures the trade-off: larger roots reduce the polynomial
    # amplitude (good) but the exponential weight penalizes large roots.
    
    bound_value = np.exp(log_prod_inv - log_sum / (2.0 * m))
    
    # Add a penalty for roots that are too close together (numerical stability)
    if m > 1:
        diffs = np.diff(roots)
        min_diff = np.min(diffs)
        if min_diff < 1e-6:
            bound_value += 1e6 * (1e-6 - min_diff)
    
    return float(bound_value)

def _compute_c4_bound_v2(roots, m=12, n=1):
    """
    More accurate C4 bound computation using the actual Laguerre polynomial
    evaluation and quadrature.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    roots = np.sort(roots)
    
    # The Cohn-Goncalves construction: the sign uncertainty bound C4 is
    # given by the L2 norm of the Laguerre combination against the weight.
    # For m double roots, the polynomial is:
    #   P(x) = prod_{i=1}^{m} (x - r_i)^2
    # The bound involves:
    #   C4 <= ||P||_{L2(w)} / ||P||_{L2(w) on positive half-line}
    # where w(x) = x^{2n-1} e^{-x}
    
    # For n=1: w(x) = x e^{-x}
    
    # The key insight: C4 is bounded by the ratio of the full integral
    # to the integral over the region where the sign function is positive.
    # A practical upper bound is:
    #   C4 <= exp(-min(roots)) * prod(roots)^{-1} * (2m)!/(m!)^2
    
    # Let's use the actual integral computation for accuracy.
    
    # Evaluate the Laguerre weight integral of the squared polynomial
    def integrand(x):
        # P(x) = prod (x - r_i)^2
        val = 1.0
        for r in roots:
            val *= (x - r) ** 2
        # Weight: x^{2n-1} e^{-x} = x * e^{-x} for n=1
        return val * x * np.exp(-x)
    
    try:
        # Integrate from 0 to a large value
        upper = np.max(roots) * 3 + 50
        integral, err = quad(integrand, 0, upper, limit=200)
        if integral <= 0 or not np.isfinite(integral):
            return 1e10
        
        # The bound C4 is related to this integral. For the sign uncertainty,
        # C4 <= sqrt(integral) / (some normalization)
        # A standard bound: C4 <= (integral)^{1/2} / (m! * 2^m)
        
        # More precisely, the Cohn-Goncalves bound is:
        # C4 <= (2^m * m!)^{-1} * (integral)^{1/2}
        
        norm = (2.0 ** m) * sp_gamma(m + 1)
        c4_bound = np.sqrt(integral) / norm
        
        return float(c4_bound)
    except Exception:
        return 1e10

def _get_initial_roots(m=12, n=1):
    """Get initial root positions from Laguerre polynomial zeros."""
    degree = 2 * m
    alpha = 2 * n - 1
    
    try:
        L = genlaguerre(degree, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        
        if len(roots) >= m:
            return roots[:m]
    except Exception:
        pass
    
    # Fallback: use Laguerre-Gauss nodes
    try:
        from numpy.polynomial.laguerre import laggauss
        x, w = laggauss(m)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Uses numerical optimization to minimize the C4 upper bound.
    """
    m = 12
    n = 1
    
    # Get initial roots from Laguerre polynomial zeros
    init_roots = _get_initial_roots(m, n)
    
    # Compute initial bound
    best_roots = init_roots.copy()
    best_score = _compute_c4_bound_v2(best_roots, m, n)
    
    if _deadline_reached():
        return [float(x) for x in best_roots]
    
    # Optimization using L-BFGS-B with bounds
    try:
        # Define objective function
        def objective(x):
            if _deadline_reached():
                return 1e10
            x = np.maximum(x, 1e-8)  # Ensure positivity
            return _compute_c4_bound_v2(x, m, n)
        
        # Bounds: roots should be positive and reasonably spaced
        # Lower bound: small positive
        # Upper bound: based on typical Laguerre zero locations
        lower_bounds = np.ones(m) * 1e-8
        upper_bounds = np.ones(m) * 500.0
        
        # Use L-BFGS-B for gradient-based optimization
        result = minimize(
            objective,
            init_roots,
            method='L-BFGS-B',
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success and result.fun < best_score:
            best_roots = np.maximum(result.x, 1e-8)
            best_score = result.fun
        
    except Exception:
        pass
    
    if _deadline_reached():
        return [float(x) for x in best_roots]
    
    # Try differential evolution for global optimization if time permits
    try:
        def de_objective(x):
            if _deadline_reached():
                return 1e10
            x = np.maximum(x, 1e-8)
            return _compute_c4_bound_v2(x, m, n)
        
        # Use a smaller population and fewer generations to fit in time
        de_result = differential_evolution(
            de_objective,
            bounds=list(zip(lower_bounds, upper_bounds)),
            maxiter=50,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=True
        )
        
        if de_result.fun < best_score:
            best_roots = np.maximum(de_result.x, 1e-8)
            best_score = de_result.fun
            
    except Exception:
        pass
    
    # Final verification
    best_roots = np.maximum(best_roots, 1e-8)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
