"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import warnings

# Precompute constants for analytic solution
# N_k = sqrt(pi) * 2^(2k) * (2k)!
# H_2k(0) = (-1)^k * (2k)! / k!

# We need coefficients for H_0, H_4, H_8, H_12
# Indices k=0, 2, 4, 6
# Let's create arrays for k=0..6

def _precompute_constants():
    N = np.zeros(7)
    H0 = np.zeros(7)
    for k in range(7):
        n = 2 * k
        # N_k = sqrt(pi) * 2^n * n!
        log_N = 0.5 * math.log(math.pi) + n * math.log(2) + math.lgamma(n + 1)
        N[k] = math.exp(log_N)
        
        # H_n(0) for even n=2k: (-1)^k * (2k)! / k!
        if k % 2 == 0:
            sign = 1.0
        else:
            sign = -1.0
        log_H0 = math.lgamma(2*k + 1) - math.lgamma(k + 1)
        H0[k] = sign * math.exp(log_H0)
    return N, H0

N_consts, H0_consts = _precompute_constants()

def _m2_analytic(c):
    """
    Compute M2^(0)(c) = ∫ x^2 (∑ c_k H_{2k}(x))^2 e^{-x^2} dx
    c is an array of length 7 (indices 0..6).
    """
    m2 = 0.0
    for i in range(7):
        if c[i] == 0: continue
        # Diagonal terms
        m2 += c[i]**2 * (2*i + 0.5) * N_consts[i]
        
        # Cross terms with i+1
        if i < 6:
            if c[i+1] != 0:
                coeff = (2*i + 2) * (2*i + 1) * N_consts[i]
                m2 += 2 * c[i] * c[i+1] * coeff
    return m2

def _norm_sq_analytic(c):
    """
    Compute || ∑ c_k H_{2k}(x) e^{-x^2/2} ||_2^2 = ∑ c_k^2 N_k
    """
    return np.sum(c**2 * N_consts)

def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) analytically.
    """
    # Construct full coefficient vector a for H_0, H_4, H_8, H_12
    # a[0] = c0, a[2] = c4, a[4] = c8, a[6] = c12
    a = np.zeros(7)
    a[0] = c0
    a[2] = c4
    a[4] = c8
    
    # Calculate a[6] (c12) such that P(0) = 0
    # ∑ a_k H_{2k}(0) = 0
    # a[6] H_{12}(0) = - (a[0]H_0(0) + a[2]H_4(0) + a[4]H_8(0))
    h0_0 = H0_consts[0]
    h4_0 = H0_consts[2]
    h8_0 = H0_consts[4]
    h12_0 = H0_consts[6]
    
    if abs(h12_0) < 1e-15:
        return 1e10
        
    a[6] = -(a[0]*h0_0 + a[2]*h4_0 + a[4]*h8_0) / h12_0
    
    # Calculate norms and moments
    norm2_g = _norm_sq_analytic(a)
    if norm2_g < 1e-15:
        return 1e10
        
    m2_g = _m2_analytic(a)
    
    # Fourier transform coefficients b_k = a_k * (-1)^k
    b = a * np.array([1, -1, 1, -1, 1, -1, 1])
    
    norm2_f = _norm_sq_analytic(b) # Same as norm2_g
    m2_f = _m2_analytic(b)
    
    # Uncertainty product
    var_g = m2_g / norm2_g
    var_f = m2_f / norm2_f
    
    return var_g * var_f

def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    try:
        bound = _compute_bound_analytic(c0, c4, c8)
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    except Exception:
        return 1e10

def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    """
    initial = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_coeffs = initial
    best_bound = _compute_bound_analytic(initial[0], initial[1], initial[2])
    
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        np.random.seed(42)
        for trial in range(10):
            if trial == 0:
                x0 = initial
            else:
                x0 = np.random.uniform([-1, -1, -1], [1, 1, 1])
            
            # Use L-BFGS-B directly as it is fast and handles bounds
            res = minimize(_objective, x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x
                
            # Also try Nelder-Mead for robustness against local minima
            res_nm = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < best_bound:
                best_bound = res_nm.fun
                best_coeffs = res_nm.x

    final_bound = _compute_bound_analytic(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if not np.isfinite(final_bound) or final_bound < 0:
        return initial
    
    return best_coeffs
# EVOLVE-BLOCK-END
