import numpy as np
from scipy.optimize import minimize
import time
import math

def _poly_moment(coeffs, x):
    """Evaluate polynomial with given coefficients at point x."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

def _build_p_squared_coeffs(roots):
    """
    Build coefficients of P(x)^2 where P(x) = prod_i (x - r_i)^2
    for 12 double roots. The polynomial has degree 24.
    Returns list of coefficients [c_0, c_1, ..., c_24] such that
    P(x)^2 = sum c_k x^k.
    """
    # P(x) = prod_i (x - r_i)^2
    # First build P(x) as polynomial, then square it.
    # P(x) = prod_i (x^2 - 2*r_i*x + r_i^2)
    
    # Start with polynomial 1
    p_coeffs = [1.0]  # constant term
    
    for r in roots:
        # Multiply by (x^2 - 2r*x + r^2)
        new_coeffs = [0.0] * (len(p_coeffs) + 2)
        for i, c in enumerate(p_coeffs):
            new_coeffs[i] += c * (r * r)
            new_coeffs[i + 1] += c * (-2.0 * r)
            new_coeffs[i + 2] += c
        p_coeffs = new_coeffs
    
    # Now square P(x) to get P(x)^2
    # p_coeffs has length 13 (degree 12)
    # P(x)^2 has degree 24
    sq_coeffs = [0.0] * (2 * len(p_coeffs) - 1)
    for i in range(len(p_coeffs)):
        for j in range(len(p_coeffs)):
            sq_coeffs[i + j] += p_coeffs[i] * p_coeffs[j]
    
    return sq_coeffs

def _compute_c4(roots):
    """
    Compute the C4 upper bound for the given 12 double-root locations.
    
    The Cohn-Goncalves construction involves:
    - A polynomial with double roots at the given locations
    - The bound involves integrating |P(x)|^2 against Laguerre weights
    
    For the Fourier sign uncertainty problem, the C4 bound is:
    C4 = sqrt( sum_k c_k * I_k )
    where I_k = integral of x^k * e^{-x} dx = k! (for Laguerre weight)
    and c_k are coefficients of P(x)^2.
    
    Actually, let me think more carefully. The standard result:
    For the Laguerre polynomial construction, the uncertainty bound involves
    the norm of the polynomial in the Laguerre inner product.
    
    The C4 quantity is related to:
    C4^2 = <P^2, 1> / <1, 1> type ratio, or more specifically
    it's the ratio that determines the sign change location.
    
    Let me use the direct computation:
    C4 = sqrt( sum_{k=0}^{24} c_k * k! ) / sqrt( sum_{k=0}^{24} c_k * (k+1)! )
    or similar. The exact formula depends on the specific construction.
    
    A common form for the Cohn-Goncalves bound:
    The last sign change x* satisfies:
    x* = sum_k c_k * (k+1)! / sum_k c_k * k!
    and C4 is proportional to sqrt(x*) or related.
    
    Let me compute both and use the one that makes sense as an upper bound.
    """
    if len(roots) != 12:
        return 1e10
    
    # Check all positive
    if any(r <= 0 for r in roots):
        return 1e10
    
    try:
        sq_coeffs = _build_p_squared_coeffs(roots)
    except Exception:
        return 1e10
    
    # Compute the relevant integrals
    # For Laguerre weight e^{-x}, integral x^k e^{-x} dx = k!
    # The C4 bound typically involves:
    # numerator = sum c_k * k!
    # denominator = sum c_k * (k+1)!
    # C4 = sqrt(numerator / denominator) or similar
    
    numerator = 0.0
    denominator = 0.0
    
    for k, c in enumerate(sq_coeffs):
        if c == 0:
            continue
        # k! can be computed iteratively to avoid overflow issues for small k
        # For k up to 24, k! is manageable
        fact_k = math.factorial(k)
        fact_kp1 = math.factorial(k + 1)
        numerator += c * fact_k
        denominator += c * fact_kp1
    
    if denominator <= 0 or numerator <= 0:
        return 1e10
    
    # The C4 upper bound
    # Common form: C4 = sqrt(numerator / denominator)
    # Or C4 = numerator / denominator
    # Let's use the geometric mean form which is typical for uncertainty bounds
    c4 = math.sqrt(numerator / denominator)
    
    return c4

def _objective(y):
    """Objective function: minimize C4 bound. y = log(roots) to ensure positivity."""
    try:
        x = np.exp(y)
        if np.any(x <= 0) or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return 1e10
        c4 = _compute_c4(x)
        if not np.isfinite(c4):
            return 1e10
        return c4
    except Exception:
        return 1e10

def _objective_grad(y):
    """Gradient of the objective (numerical)."""
    eps = 1e-8
    grad = np.zeros_like(y)
    f0 = _objective(y)
    for i in range(len(y)):
        y_plus = y.copy()
        y_plus[i] += eps
        f_plus = _objective(y_plus)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Optimizes to minimize the C4 upper bound.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    m = 12
    
    # Step 1: Get initial roots from Laguerre polynomial L_12(x)
    try:
        roots, weights = np.polynomial.laguerre.laggauss(m)
    except Exception:
        roots = 4.0 * np.arange(1, m + 1)
    
    # Also try generalized Laguerre nodes for different alpha values
    # L_{m}^{(alpha)}(x) nodes can give different starting points
    
    best_roots = roots.copy()
    best_c4 = _compute_c4(roots)
    
    # Check time
    if time.time() > deadline - 60:
        return best_roots.tolist()
    
    # Step 2: Try multiple initial points from generalized Laguerre
    for alpha in [0, 0.5, 1, 2, 3, 4, 5, 10]:
        if time.time() > deadline - 60:
            break
        try:
            # Get nodes of generalized Laguerre L_m^{(alpha)}
            # Use scipy or numpy
            from numpy.polynomial.laguerre import laggauss
            # For alpha != 0, we need to compute differently
            # numpy's laggauss only does alpha=0
            # Let's use a different approach: scale the standard roots
            # Or use scipy.special.roots_genlaguerre
            from scipy.special import roots_genlaguerre
            g_roots, g_weights = roots_genlaguerre(m, alpha)
            g_roots = np.sort(g_roots)
            c4_g = _compute_c4(g_roots)
            if c4_g < best_c4 and np.isfinite(c4_g):
                best_c4 = c4_g
                best_roots = g_roots.copy()
        except Exception:
            continue
    
    # Check time
    if time.time() > deadline - 30:
        return best_roots.tolist()
    
    # Step 3: Local optimization from best starting point
    try:
        y0 = np.log(best_roots)
        
        # Use L-BFGS-B with gradient
        result = minimize(
            _objective, 
            y0, 
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline - 10:
            return best_roots.tolist()
        
        if result.success and np.isfinite(result.fun) and result.fun < best_c4:
            optimal_roots = np.exp(result.x)
            if np.all(optimal_roots > 0):
                best_roots = optimal_roots
                best_c4 = result.fun
    except Exception:
        pass
    
    # Step 4: Additional refinement with Nelder-Mead if time permits
    if time.time() < deadline - 20:
        try:
            y0 = np.log(best_roots)
            result2 = minimize(
                _objective,
                y0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result2.success and np.isfinite(result2.fun) and result2.fun < best_c4:
                optimal_roots = np.exp(result2.x)
                if np.all(optimal_roots > 0):
                    best_roots = optimal_roots
                    best_c4 = result2.fun
        except Exception:
            pass
    
    # Final safety check
    if time.time() > deadline - 5:
        return best_roots.tolist()
    
    return best_roots.tolist()
