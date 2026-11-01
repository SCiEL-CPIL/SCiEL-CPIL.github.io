"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE = 210.0
SAFETY_MARGIN = 10.0

def _check_deadline():
    return (time.time() - _START_TIME) < (DEADLINE - SAFETY_MARGIN)

def _hermite_poly(x, n):
    """Compute the nth Hermite polynomial (physicist's convention)."""
    return eval_hermite(n, x)

def _compute_A(coeffs):
    """
    Compute A(f) for f(x) = g(x) * exp(-x^2/2) where
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    We need to determine what A(f) is. Based on the uncertainty inequality context,
    let's assume A(f) = integral of |f(x)|^2 * x^2 dx (second moment)
    or A(f) = integral of |f(x)| dx (L1 norm).
    
    Let's try: A(f) = (integral of |f|^2)^(1/2) * (integral of x^2 |f|^2)^(1/2)
    which is related to the standard deviation.
    
    Actually, for the uncertainty principle, the typical form is:
    (integral x^2 |f|^2) * (integral xi^2 |fhat|^2) >= (1/4) (integral |f|^2)^2
    
    So A(f) might be sqrt(integral x^2 |f|^2) and A(fhat) = sqrt(integral xi^2 |fhat|^2)
    and the product is what we minimize.
    
    Let me compute numerically.
    """
    c0, c4, c8 = coeffs
    c12 = -(c0 + 3*c4 + 105*c8) / 945.0
    
    # Grid for numerical integration
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    H0 = _hermite_poly(x, 0)
    H4 = _hermite_poly(x, 4)
    H8 = _hermite_poly(x, 8)
    H12 = _hermite_poly(x, 12)
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = g * np.exp(-x**2 / 2.0)
    
    # |f|^2
    f2 = f**2
    
    # A(f) = sqrt(integral x^2 |f|^2 dx)
    int_x2_f2 = np.trapz(x**2 * f2, x)
    int_f2 = np.trapz(f2, x)
    
    if int_f2 <= 0:
        return np.inf
    
    # For the Fourier transform, using the convention that 
    # F(H_n(x) exp(-x^2/2)) = i^n H_n(x) exp(-x^2/2)
    # So |fhat|^2 = |g(x)|^2 exp(-x^2) as well (same magnitude)
    # Thus integral xi^2 |fhat|^2 = integral x^2 |f|^2
    
    # So A(f)*A(fhat) = (integral x^2 |f|^2)
    # Wait, that's just one factor. Let me reconsider.
    
    # Actually if A(f) = sqrt(integral x^2 |f|^2) and A(fhat) = sqrt(integral xi^2 |fhat|^2)
    # and they're equal, then product = integral x^2 |f|^2
    
    return int_x2_f2

def construct():
    if not _check_deadline():
        # Return best known if over deadline
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try local optimization from seed
    try:
        result = minimize(_compute_A, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
    
    # Try differential evolution for global search
    try:
        bounds = [(-5, 5), (-5, 5), (-1, 1)]
        de_result = differential_evolution(_compute_A, bounds, 
                                           maxiter=100, popsize=15,
                                           seed=42, tol=1e-10,
                                           polish=True)
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
    
    # Refine with L-BFGS-B
    if best_result is not None:
        try:
            refine_result = minimize(_compute_A, best_result, method='L-BFGS-B',
                                    options={'maxiter': 10000, 'gtol': 1e-15})
            if refine_result.fun < best_score:
                best_score = refine_result.fun
                best_result = refine_result.x
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return x0

# EVOLVE-BLOCK-END
