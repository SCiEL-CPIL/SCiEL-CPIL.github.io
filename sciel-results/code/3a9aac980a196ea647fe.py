import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

def _compute_uncertainty_analytic(a0, a4, a8):
    """Compute A(f)*A(Fourier(f)) analytically."""
    if time.time() > _DEADLINE:
        return None
    
    # Hermite values at 0
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 945.0
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # N_k = sqrt(pi) * 2^k * k!
    # We only need ratios, so sqrt(pi) cancels.
    # N_k proportional to 2^k * k!
    def N(k):
        return (2 ** k) * np.math.factorial(k)
    
    num = 0.0
    den = 0.0
    for c, k in zip(coeffs, degrees):
        nk = N(k)
        num += (c * c) * nk * (k + 0.5)
        den += (c * c) * nk
    
    if den <= 0:
        return 1e10
        
    return num / den

def construct():
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_uncertainty_analytic(*seed)
    if best_val is None:
        return seed
    best_coeffs = seed.copy()
    
    def objective(params):
        val = _compute_uncertainty_analytic(*params)
        if val is None:
            return 1e10
        return val
    
    # Try to optimize
    rng = np.random.default_rng(42)
    start_time = time.time()
    max_time = 180
    
    # Use a grid of random restarts
    for restart in range(100):
        if time.time() - start_time > max_time - 5:
            break
        
        # Random perturbation
        # Scale decreases as we go? Or fixed?
        # Let's try a mix of local and global
        if restart < 20:
            x0 = best_coeffs + rng.normal(0, 0.1, 3)
        else:
            x0 = rng.normal(0, 1.0, 3)
            
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
            
    return best_coeffs
