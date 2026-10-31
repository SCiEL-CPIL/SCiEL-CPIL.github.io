"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

# Deadline guard: kill after 210s wall-clock, return best with safety margin
_START_TIME = time.time()
_TIME_LIMIT = 205.0  # 5s safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _check_deadline():
    if _time_left() < 2.0:
        return True
    return False

# Hermite polynomials (physicists' convention) H_n(x)
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
# H_12 = 4096x^12 - 73728x^10 + 516096x^8 - 1651584x^6 + 2478096x^4 - 1612800x^2 + 103950

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return 16*x**4 - 48*x**2 + 12

def _H8(x):
    return 256*x**8 - 3072*x**6 + 10752*x**4 - 10752*x**2 + 1680

def _H12(x):
    return (4096*x**12 - 73728*x**10 + 516096*x**8 
            - 1651584*x**6 + 2478096*x**4 
            - 1612800*x**2 + 103950)

def _gaussian(x, a=1.0):
    """Gaussian envelope exp(-a*x^2)"""
    return np.exp(-a * x**2)

def _build_g(coeffs, x):
    """Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
       where c12 is derived so that g(0) = 0."""
    c0, c4, c8 = coeffs
    # At x=0: H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=103950
    # g(0) = (c0*1 + c4*12 + c8*1680 + c12*103950) * exp(0) = 0
    # => c12 = -(c0 + 12*c4 + 1680*c8) / 103950
    c12 = -(c0 + 12*c4 + 1680*c8) / 103950.0
    return (c0*_H0(x) + c4*_H4(x) + c8*_H8(x) + c12*_H12(x)) * _gaussian(x, 0.5)

def _compute_bound(coeffs):
    """Compute the upper bound on A(f)*A(F(f)) = sigma_x * sigma_p.
    
    For g(x) = P(x) * exp(-x^2/2), we compute:
    - sigma_x^2 = <x^2> - <x>^2 (position variance)
    - sigma_p^2 = <(-i d/dx)^2> - <(-i d/dx)>^2 (momentum variance)
    
    Using numerical integration on a fine grid."""
    x = np.linspace(-30, 30, 20001)
    dx = x[1] - x[0]
    g = _build_g(coeffs, x)
    
    # Check if g is essentially zero
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-30:
        return 1e10  # Degenerate
    
    # Normalize
    g_norm = g / np.sqrt(norm_sq)
    
    # <x>
    x_mean = np.sum(x * g_norm**2) * dx
    # <x^2>
    x2_mean = np.sum(x**2 * g_norm**2) * dx
    sigma_x2 = x2_mean - x_mean**2
    
    if sigma_x2 < 1e-30:
        return 1e10
    
    # Derivative g'(x) for momentum
    # g(x) = P(x) * exp(-x^2/2)
    # g'(x) = P'(x)*exp(-x^2/2) + P(x)*(-x)*exp(-x^2/2) = [P'(x) - x*P(x)]*exp(-x^2/2)
    c0, c4, c8 = coeffs
    c12 = -(c0 + 12*c4 + 1680*c8) / 103950.0
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P'(x) = c4*H4' + c8*H8' + c12*H12'  (H0'=0)
    # H4'(x) = 64x^3 - 96x
    # H8'(x) = 2048x^7 - 18432x^5 + 43008x^3 - 21504x
    # H12'(x) = 49152x^11 - 737280x^9 + 4128768x^7 - 9909504x^5 + 9912384x^3 - 3225600x
    
    H4_prime = 64*x**3 - 96*x
    H8_prime = 2048*x**7 - 18432*x**5 + 43008*x**3 - 21504*x
    H12_prime = (49152*x**11 - 737280*x**9 + 4128768*x**7 
                 - 9909504*x**5 + 9912384*x**3 - 3225600*x)
    
    P = c0 + c4*_H4(x) + c8*_H8(x) + c12*_H12(x)
    P_prime = c4*H4_prime + c8*H8_prime + c12*H12_prime
    
    # g'(x) = [P'(x) - x*P(x)] * exp(-x^2/2)
    g_prime = (P_prime - x*P) * _gaussian(x, 0.5)
    
    # <p> where p = -i d/dx, so <p> = -i * integral g* g' dx (real part is 0 for real g)
    # For real g, <p> = 0 (odd integrand)
    # <p^2> = integral |g'|^2 dx
    p2_mean = np.sum(g_prime**2) * dx
    
    # sigma_p^2 = <p^2> - <p>^2 = <p^2> (since <p>=0 for real g)
    sigma_p2 = p2_mean
    
    if sigma_p2 < 1e-30:
        return 1e10
    
    sigma_x = np.sqrt(sigma_x2)
    sigma_p = np.sqrt(sigma_p2)
    
    return sigma_x * sigma_p

def _objective(coeffs):
    """Objective: upper bound on A(f)*A(F(f))"""
    return _compute_bound(coeffs)

def _objective_with_grad(coeffs):
    """Objective with numerical gradient."""
    f = _compute_bound(coeffs)
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        cp = coeffs.copy()
        cm = coeffs.copy()
        cp[i] += eps
        cm[i] -= eps
        grad[i] = (_compute_bound(cp) - _compute_bound(cm)) / (2*eps)
    return f, grad

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0."""
    
    # Prior state of the art
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_bound(best_coeffs)
    
    if _check_deadline():
        return best_coeffs
    
    # Strategy: Constrained optimization using L-BFGS-B with multiple starts
    # The problem: minimize sigma_x * sigma_p subject to g(0)=0 (already built in)
    # and normalization (handled in _compute_bound)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [best_coeffs.copy()]
    
    # Random starts in a reasonable range
    for _ in range(30):
        c0 = rng.uniform(-3, 3)
        c4 = rng.uniform(-0.1, 0.1)
        c8 = rng.uniform(-0.001, 0.001)
        starts.append(np.array([c0, c4, c8]))
    
    # Grid-based starts around the known solution
    for c0 in np.linspace(-2, 2, 7):
        for c4 in np.linspace(-0.05, 0.05, 7):
            if _check_deadline():
                break
            c8 = 1/3240.0  # Keep c8 near known value
            starts.append(np.array([c0, c4, c8]))
        if _check_deadline():
            break
    
    # Also try different scales
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        c0 = rng.uniform(-scale, scale)
        c4 = rng.uniform(-scale*0.01, scale*0.01)
        c8 = rng.uniform(-scale*0.001, scale*0.001)
        starts.append(np.array([c0, c4, c8]))
        if _check_deadline():
            break
    
    # Optimize from each start
    for start in starts:
        if _check_deadline():
            break
        
        # Evaluate at start
        val = _compute_bound(start)
        if val < best_val:
            best_val = val
            best_coeffs = start.copy()
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                lambda c: _compute_bound(c),
                start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_val:
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
        
        if _check_deadline():
            break
    
    # Refine with more iterations from the best found
    if _time_left() > 5.0:
        try:
            result = minimize(
                lambda c: _compute_bound(c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final local refinement using Nelder-Mead for robustness
    if _time_left() > 5.0:
        try:
            result = minimize(
                lambda c: _compute_bound(c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
