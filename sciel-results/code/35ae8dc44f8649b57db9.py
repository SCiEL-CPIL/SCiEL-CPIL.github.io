"""Optimized program for uncertainty inequality coefficient search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import roots_hermite

def _get_hermite_values(n, x):
    """
    Compute the nth physicists' Hermite polynomial H_n(x) and its derivative H_n'(x)
    for an array x.
    H_n'(x) = 2n H_{n-1}(x)
    """
    if n == 0:
        Hn = np.ones_like(x)
        Hn_prime = np.zeros_like(x)
        return Hn, Hn_prime
    elif n == 1:
        Hn = 2 * x
        Hn_prime = 2 * np.ones_like(x)
        return Hn, Hn_prime
    
    H0 = np.ones_like(x)
    H1 = 2 * x
    H0_prime = np.zeros_like(x)
    H1_prime = 2 * np.ones_like(x)
    
    for k in range(2, n + 1):
        H2 = 2 * x * H1 - 2 * (k - 1) * H0
        H2_prime = 2 * k * H1
        
        H0, H1 = H1, H2
        H0_prime, H1_prime = H1_prime, H2_prime
        
    return H1, H1_prime

def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a, b, c].
    The H12 coefficient d is fixed by g(0) = 0.
    g(x) = (a H0 + b H4 + c H8 + d H12) * exp(-x^2/2)
    """
    a, b, c = coeffs
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    if not hasattr(_compute_bound, 'quad_data'):
        n_quad = 60
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0, dH0 = _get_hermite_values(0, x_quad)
        H4, dH4 = _get_hermite_values(4, x_quad)
        H8, dH8 = _get_hermite_values(8, x_quad)
        H12, dH12 = _get_hermite_values(12, x_quad)
        
        _compute_bound.quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
        
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _compute_bound.quad_data
    
    P = a * H0 + b * H4 + c * H8 + d * H12
    dP = a * dH0 + b * dH4 + c * dH8 + d * dH12
    
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15:
        return 1e10
        
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    return bound

def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Define bounds for coefficients.
    # The problem is homogeneous in scale, so only ratios matter.
    # We can constrain one coefficient to be 1 or restrict the vector norm.
    # Differential evolution works best with bounded variables.
    # Let's search in a box [-10, 10] for a, b, c.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Initial population to guide DE
    # We know that [1, 0, 0] is a valid starting point (g ~ H0 - small H12).
    # Let's initialize the population with some structured guesses.
    # DE in scipy allows `init` as an array of shape (popsize, n).
    # But we must ensure it's within bounds.
    
    # Let's just let DE do its thing, but with a decent popsize and maxiter.
    # Time budget is 180s.
    
    try:
        # Use differential_evolution for global optimization
        result = differential_evolution(
            _compute_bound,
            bounds=bounds,
            maxiter=1000,
            popsize=15,
            tol=1e-8,
            atol=1e-12,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        
        if result.success or result.fun < 1e10:
            best_coeffs = result.x
        else:
            best_coeffs = np.array([1.0, 0.0, 0.0])
            
    except Exception:
        best_coeffs = np.array([1.0, 0.0, 0.0])
        
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
