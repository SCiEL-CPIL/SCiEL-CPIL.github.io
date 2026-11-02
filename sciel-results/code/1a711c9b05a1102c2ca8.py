"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

_DEADLINE = 175  # safety margin before 210s kill

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        # H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 12:
        # H_12(x) = 4096x^12 - 147456x^10 + 2580480x^8 - 20321280x^6 + 81729600x^4 - 172972800x^2 + 103219200
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

# Precompute quadrature data for fast evaluation
_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 100
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        # Derivatives: H_n'(x) = 2n H_{n-1}(x)
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
    return _quad_data

def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a0, a4, a8].
    The H12 coefficient d is fixed by g(0) = 0.
    g(x) = (a0 H0 + a4 H4 + a8 H8 + d H12) * exp(-x^2/2)
    
    We minimize sigma_x * sigma_k where:
    sigma_x^2 = <x^2> - <x>^2 (but <x>=0 by symmetry)
    sigma_k^2 = <k^2> in Fourier domain = <g'^2> / <g^2>
    """
    if time.time() > _DEADLINE:
        return float('inf')
    
    a0, a4, a8 = coeffs
    
    # H_n(0) values for physicists' Hermite polynomials
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 40320  (from 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320)
    # H_12(0) = 103219200 (from the polynomial above)
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 40320.0
    H12_0 = 103219200.0
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + d*H12(0) = 0
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _get_quad_data()
    
    # Construct P(x) and P'(x)
    P = a0 * H0 + a4 * H4 + a8 * H8 + d * H12
    dP = a0 * dH0 + a4 * dH4 + a8 * dH8 + d * dH12
    
    # g(x) = P(x) * exp(-x^2/2)
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    # With Hermite quadrature: sum(w_i * f(x_i)) approximates (1/sqrt(pi)) * integral(f(x) * exp(-x^2) dx)
    # So the ratio of integrals cancels the 1/sqrt(pi) factor
    
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15:
        return 1e10
        
    # sigma_x^2 = <x^2> = sum(w * x^2 * P^2) / sum(w * P^2)
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    # g'(x) = (P'(x) - x P(x)) * exp(-x^2/2)
    # |g'(x)|^2 = (P'(x) - x P(x))^2 * exp(-x^2)
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    # sigma_k^2 = <k^2> = (1/2pi) * integral(|g'(x)|^2 dx) / (1/2pi) * integral(|g(x)|^2 dx)
    #           = integral(|g'(x)|^2 dx) / integral(|g(x)|^2 dx)
    # Using Hermite quadrature:
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    return bound

def construct():
    start_time = time.time()
    
    # The constraint g(0) = 0 fixes a12 in terms of a0, a4, a8
    # H0(0) = 1, H4(0) = 12, H8(0) = 40320, H12(0) = 103219200
    # So: a0 + 12*a4 + 40320*a8 + 103219200*a12 = 0
    # => a12 = -(a0 + 12*a4 + 40320*a8) / 103219200
    
    def objective(params):
        if time.time() > _DEADLINE:
            return float('inf')
        return _compute_bound(params)
    
    # Start from the seed solution
    seed = np.array([-113/100, 1/25, 1/3240])
    
    best_result = seed.copy()
    best_score = _compute_bound(seed)
    
    # Try local optimization from seed
    try:
        result = minimize(objective, seed, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x.copy()
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < _DEADLINE - 30:
        for _ in range(10):
            if time.time() > _DEADLINE - 30:
                break
            try:
                random_start = np.random.uniform(-2, 2, 3)
                result = minimize(objective, random_start, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish pass
    if time.time() < _DEADLINE - 10:
        try:
            result = minimize(objective, best_result, method='Nelder-Mead', 
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
