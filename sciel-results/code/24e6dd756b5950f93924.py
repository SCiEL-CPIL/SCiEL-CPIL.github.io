"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound C4 (variance product) for given Hermite coefficients.
    
    The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2)
    where a12 is chosen so that g(0) = 0.
    
    The bound is proportional to (variance of x)^2. We minimize this variance.
    """
    # Values of H_n(0) for physicists' Hermite polynomials
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    
    # Constraint g(0) = 0 implies P(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Precompute normalization factors N_n = sqrt(pi * 2^n * n!)
    N_0 = np.sqrt(np.pi)
    N_4 = np.sqrt(384 * np.pi)
    N_8 = np.sqrt(10321920 * np.pi)
    N_12 = np.sqrt(4096 * 479001600 * np.pi)
    
    c_0 = a0 * N_0
    c_2 = a4 * N_4
    c_4 = a8 * N_8
    c_6 = a12 * N_12
    
    c = np.array([c_0, c_2, c_4, c_6])
    
    norm_sq = np.sum(c**2)
    if norm_sq < 1e-12:
        return 1e10
        
    # Compute <x^2>_g
    x2_expect = 0.0
    
    for i in range(4):
        n_i = 2 * i
        x2_expect += c[i]**2 * (n_i + 0.5)
        
        for j in range(i + 1, 4):
            n_j = 2 * j
            if abs(n_i - n_j) == 2:
                n_low = min(n_i, n_j)
                val = 0.5 * np.sqrt((n_low + 1) * (n_low + 2))
                x2_expect += 2 * c[i] * c[j] * val
    
    var_x = x2_expect / norm_sq
    
    # The product is proportional to var_x^2.
    return var_x * var_x

def construct():
    """
    Find optimal Hermite coefficients to minimize the upper bound C4.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s kill

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    # 1. Multi-start Nelder-Mead with wider exploration
    np.random.seed(42)
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x_start = best_x.copy()
        else:
            # Random perturbation with varying scales
            if i < 5:
                scale = 0.1
            else:
                scale = 1.0
            
            x_start = best_x + np.random.randn(3) * scale
            # Ensure we don't go to crazy values
            x_start[0] = np.clip(x_start[0], -100, 100)
            x_start[1] = np.clip(x_start[1], -10, 10)
            x_start[2] = np.clip(x_start[2], -1, 1)
        
        try:
            result = minimize(
                _compute_bound,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 20:
            break

    # 2. High-precision Polish with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if np.isfinite(result2.fun) and result2.fun < best_bound:
                best_bound = result2.fun
                best_x = result2.x.copy()
        except Exception:
            pass

    # 3. Final check
    if time.time() < deadline - 1:
        final_val = _compute_bound(*best_x)
        if np.isfinite(final_val) and final_val < best_bound:
            best_bound = final_val

    return best_x
# EVOLVE-BLOCK-END
