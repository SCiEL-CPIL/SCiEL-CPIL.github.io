"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = _START + 170.0  # safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# Pre-compute constants for analytic evaluation
# Indices in basis: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Norms of Hermite polynomials: <H_n, H_n> = sqrt(pi) * 2^n * n!
norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# Diagonal elements of <x^2 H_i, H_i> = (i + 0.5) * <H_i, H_i>
x2_diag = np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)])

# Values of H_n(0) for n in indices to determine c12
# H_0(0) = 1
# H_4(0) = 12
# H_8(0) = 1680
# H_12(0) = 665280
H_vals_at_0 = np.array([1.0, 12.0, 1680.0, 665280.0])

# We need the norms for the odd Hermite polynomials that appear in Q(x) = P' - xP
# Q(x) coefficients d_k for H_k where k is odd: 1, 3, 5, 7, 9, 11, 13
odd_indices = [1, 3, 5, 7, 9, 11, 13]
odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])

def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product bound A(f)*A(F) analytically.
    """
    # c12 is determined by g(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * H_vals_at_0[0] + c4 * H_vals_at_0[1] + c8 * H_vals_at_0[2]) / H_vals_at_0[3]
    
    coeffs = np.array([c0, c4, c8, c12])
    
    # 1. Calculate Norm N = sum c_m^2 * <H_m, H_m>
    norm_sq = np.sum(coeffs**2 * norms)
    if norm_sq < 1e-15:
        return 1e15
        
    # 2. Calculate <x^2> numerator = sum c_m^2 * (m+0.5) * <H_m, H_m>
    x2_num = np.sum(coeffs**2 * x2_diag)
    
    # 3. Calculate <p^2> numerator = sum d_k^2 * <H_k, H_k> for odd k
    # d_k are coefficients of H_k in Q(x) = P'(x) - x P(x)
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # Q(x) = c0(-0.5 H1) + c4(4 H3 - 0.5 H5) + c8(8 H7 - 0.5 H9) + c12(12 H11 - 0.5 H13)
    
    d1 = -0.5 * c0
    d3 = 4.0 * c4
    d5 = -0.5 * c4
    d7 = 8.0 * c8
    d9 = -0.5 * c8
    d11 = 12.0 * c12
    d13 = -0.5 * c12
    
    d_coeffs = np.array([d1, d3, d5, d7, d9, d11, d13])
    p2_num = np.sum(d_coeffs**2 * odd_norms)
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 by symmetry.
    delta_x_sq = x2_num / norm_sq
    delta_p_sq = p2_num / norm_sq
    
    # Uncertainty product
    return np.sqrt(delta_x_sq * delta_p_sq)

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
_RAW_SEED = _compute_bound_analytic(*_SEED)

def _objective(x):
    return _compute_bound_analytic(x[0], x[1], x[2])

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _compute_bound_analytic(*_SEED)
    
    # Bounds for differential evolution
    # The seed is roughly [-1.13, 0.04, 0.0003]
    # Let's try a wide range first
    bounds = [(-10, 10), (-10, 10), (-5, 5)]
    
    try:
        # 1. Differential Evolution for global search
        if _time_left() > 60:
            res = differential_evolution(
                _objective, bounds, seed=42, maxiter=300, popsize=30,
                tol=1e-10, polish=False, init='sobol',
                update_interval=1, 
                callback=lambda xk, convergence: convergence and (time.time() > _DEADLINE - 120)
            )
            if res.x is not None and res.fun < best_v:
                best_v = res.fun
                best_x = res.x
        
        # 2. Local polish with Nelder-Mead
        if _time_left() > 20:
            for x0 in [best_x, _SEED]:
                try:
                    r2 = minimize(_objective, x0, method='Nelder-Mead',
                                  options={'maxiter':5000, 'xatol':1e-12, 'fatol':1e-14, 'adaptive':True})
                    if r2.fun < best_v:
                        best_v = r2.fun
                        best_x = r2.x
                except Exception:
                    pass
        
        # 3. Multi-start DE if time permits
        if _time_left() > 60:
            for s in [1, 2, 3]:
                if _time_left() < 40: break
                try:
                    res = differential_evolution(_objective, bounds, seed=s, maxiter=200, popsize=25, tol=1e-10, polish=False)
                    if res.fun < best_v:
                        best_v = res.fun
                        best_x = res.x
                except Exception:
                    pass
            
            # Final polish
            if _time_left() > 5:
                try:
                    r2 = minimize(_objective, best_x, method='L-BFGS-B', options={'maxiter':10000, 'gtol':1e-14})
                    if r2.fun < best_v:
                        best_v = r2.fun
                        best_x = r2.x
                except Exception:
                    pass
                    
    except Exception:
        pass
        
    return np.array([best_x[0], best_x[1], best_x[2]])
# EVOLVE-BLOCK-END
