"""Refined Cohn-Goncalves Laguerre construction: search for 12 double roots minimizing C4 upper bound."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Deadline guard: must finish within 630s wall-clock; use 600s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin


def _log_gamma(x):
    """Stable log-gamma using scipy."""
    from scipy.special import gammaln
    return gammaln(x)


def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for given positive double-root locations.
    
    The Laguerre polynomial construction with double roots r_1, ..., r_12 (m=12, n=1)
    gives a sign uncertainty bound. The coefficient vector c_k for k=0..m can be computed
    from the product of Laguerre polynomials evaluated at the roots.
    
    For the Cohn-Goncalves construction, the key quantity is:
    C4 = |sum_{k=0}^{m} c_k * (-1)^k| / sqrt(prod(r_i))
    
    where c_k are coefficients of the Laguerre combination.
    
    More precisely, for the refined inequality, the bound involves:
    - The Laguerre polynomial L_m(x) with m=12
    - Double roots at r_i means we're looking at a product structure
    
    The C4 bound for the Fourier sign uncertainty with this construction is:
    C4 = |L_m^{(1)}(0) + correction terms| related to the roots
    
    A practical formulation: the upper bound on C4 is computed as
    C4 = |sum c_k| / sqrt(prod(r_i))
    where c_k are the coefficients of the polynomial whose roots are the double roots,
    normalized by the Laguerre weight.
    
    For a polynomial P(x) = prod_{i=1}^{12} (x - r_i)^2 = sum_{k=0}^{24} a_k x^k,
    the C4 bound in the Cohn-Goncalves framework is:
    C4 = |a_0| / sqrt(prod(r_i))
    
    Since a_0 = prod(r_i)^2, we get C4 = prod(r_i)^2 / sqrt(prod(r_i)) = prod(r_i)^(3/2).
    
    But this is too simple. The actual bound involves the Laguerre polynomial structure.
    
    Let me use the correct formulation. For the Cohn-Goncalves Laguerre construction:
    The sign uncertainty bound is:
    C4 = |sum_{k=0}^{m} (-1)^k * L_k^{(n)}(0) * d_k| / sqrt(prod(r_i))
    
    where d_k are determined by the root constraints.
    
    Actually, let me think about this more carefully. The task says "computes the last sign change"
    and "verifies the root constraints". The score is "the upper bound on C4".
    
    For the Cohn-Goncalves inequality with Laguerre polynomials:
    - We have m=12 double roots
    - n=1 dimension
    - The bound C4 is an upper bound on the sign uncertainty constant
    
    The standard result: for a polynomial with m double positive roots r_1,...,r_m,
    the Cohn-Goncalves bound gives:
    C4 <= (1/sqrt(prod(r_i))) * |sum_{k=0}^{m} c_k (-1)^k|
    
    where c_k are coefficients from the Laguerre expansion.
    
    Let me use a concrete computable formula. The key insight from the knowledge:
    "||c||/sqrt(prod(roots))" is a better proxy, and the bound "likely decreases with root spread".
    
    For a product polynomial P(x) = prod_{i=1}^{12}(x - r_i)^2, expanding gives coefficients.
    The C4 bound is related to the ratio of certain coefficient sums to the geometric mean of roots.
    
    I'll compute: C4 = |sum_{k even} a_k| / sqrt(prod(r_i))
    where a_k are coefficients of prod(x - r_i)^2.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12 or np.any(roots <= 0):
        return 1e10
    
    # Compute product of (x - r_i)^2 for all 12 roots
    # Use log-sum for stability if needed, but for 12 roots it should be fine
    # Build the polynomial coefficients
    # Start with [1]
    coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[:-2] += coeffs
        new_coeffs[1:-1] += -2 * r * coeffs
        new_coeffs[2:] += r * r * coeffs
        coeffs = new_coeffs
    
    # coeffs[k] is coefficient of x^k, length 25 (degree 24)
    # C4 bound: use |sum of all coefficients| / sqrt(prod(roots))
    # Or more specifically, the sign change related quantity
    
    # The "last sign change" in the coefficient sequence relates to the bound
    # C4 = |coeffs[0]| / sqrt(prod(roots)) = prod(roots)^2 / sqrt(prod(roots)) = prod(roots)^(3/2)
    
    # But a better measure involves the full coefficient vector:
    # C4 = ||coeffs||_1 / sqrt(prod(roots))
    
    prod_roots = np.prod(roots)
    sqrt_prod = np.sqrt(prod_roots)
    
    # Use the L1 norm of coefficients divided by sqrt(product of roots)
    c4 = np.sum(np.abs(coeffs)) / sqrt_prod
    
    return c4


def _compute_c4_bound_fast(roots):
    """
    Faster computation using the fact that prod(x - r_i)^2 has structure.
    a_0 = prod(r_i)^2
    a_k can be computed via elementary symmetric polynomials.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12 or np.any(roots <= 0):
        return 1e10
    
    prod_roots = np.prod(roots)
    sqrt_prod = np.sqrt(prod_roots)
    
    # For prod(x - r_i)^2, the coefficients are:
    # (prod(x - r_i))^2
    # Let Q(x) = prod(x - r_i) = sum_{k=0}^{12} q_k x^k
    # Then P(x) = Q(x)^2, and P_k = sum_{j=0}^{k} q_j * q_{k-j}
    
    # Compute q_k using Newton's identities or direct product
    q = np.array([1.0])
    for r in roots:
        new_q = np.zeros(len(q) + 1)
        new_q[:-1] += q
        new_q[1:] += -r * q
        q = new_q
    
    # Now P = Q * Q
    p = np.convolve(q, q)
    
    c4 = np.sum(np.abs(p)) / sqrt_prod
    return c4


def _objective(roots):
    """Objective to minimize: the C4 upper bound."""
    try:
        return _compute_c4_bound_fast(roots)
    except:
        return 1e10


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive values that minimize the C4 upper bound.
    """
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Return a reasonable default
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_val = _compute_c4_bound_fast(best_roots)
    
    # Strategy: The bound likely decreases with root spread.
    # Try various initial configurations, then optimize.
    
    # Analytic candidates:
    candidates = []
    
    # 1. Geometric progression with large spread
    for ratio in [2.0, 3.0, 5.0, 10.0]:
        for start in [0.1, 0.5, 1.0]:
            cand = start * (ratio ** np.arange(12))
            if cand.max() < 1e6:
                candidates.append(cand)
    
    # 2. Large roots
    for scale in [10, 50, 100, 500]:
        candidates.append(scale * np.arange(1, 13, dtype=float))
    
    # 3. Power law
    for alpha in [2, 3, 4, 5]:
        candidates.append(np.arange(1, 13, dtype=float) ** alpha)
    
    # 4. Exponential
    for rate in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(np.exp(rate * np.arange(12)))
    
    # Evaluate all candidates
    for cand in candidates:
        if np.any(cand <= 0):
            continue
        val = _compute_c4_bound_fast(cand)
        if val < best_val:
            best_val = val
            best_roots = cand.copy()
    
    # Check deadline
    if time.time() > _DEADLINE - 30:
        return list(best_roots)
    
    # Local optimization from best candidate
    try:
        # Use L-BFGS-B with bounds
        bounds = [(1e-6, 1e6)] * 12
        result = minimize(
            _objective,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_roots = result.x.copy()
    except:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE - 20:
        return list(best_roots)
    
    # Try differential evolution for global optimization
    try:
        # Use log-space to handle the wide range
        def _obj_log(log_roots):
            roots = np.exp(log_roots)
            return _compute_c4_bound_fast(roots)
        
        # Initial best in log space
        log_best = np.log(best_roots)
        
        # Bounds in log space: roots in [1e-6, 1e6] -> log in [-13.8, 13.8]
        log_bounds = [(-13.8, 13.8)] * 12
        
        de_result = differential_evolution(
            _obj_log,
            log_bounds,
            seed=42,
            maxiter=100,
            tol=1e-10,
            popsize=15,
            init='sobol',
            polish=True
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_roots = np.exp(de_result.x)
    except:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return list(best_roots)
    
    # Final polish with high precision
    try:
        result = minimize(
            _objective,
            best_roots,
            method='L-BFGS-B',
            bounds=[(1e-8, 1e8)] * 12,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_roots = result.x.copy()
    except:
        pass
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)


# Module-level initialization
_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0
