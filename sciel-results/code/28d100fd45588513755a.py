import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite
from scipy.integrate import quad

# Constants for Hermite polynomials at 0
# H_0(0) = 1
# H_4(0) = -3
# H_8(0) = 105
# H_12(0) = -10395
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

def compute_coeff_a12(a0, a4, a8):
    # a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a0 - 3a4 + 105a8 - 10395 a12 = 0
    return (a0 - 3.0*a4 + 105.0*a8) / 10395.0

def objective(coeffs):
    a0, a4, a8 = coeffs
    a12 = compute_coeff_a12(a0, a4, a8)
    
    # Coefficients for the sum
    # g(x) = exp(-pi x^2) * sum(ak * Hk(x))
    # Fhat(xi) = exp(-pi xi^2) * sum(ak * (-i)^k * Hk(xi))
    
    # We integrate |g(x)| and |Fhat(xi)| from -inf to inf.
    # Due to symmetry and Gaussian decay, integrate from 0 to L and multiply by 2?
    # No, H_n(x) parity: H_n(-x) = (-1)^n H_n(x).
    # g(x) is even because all n are even.
    # Fhat(xi) is even because (-i)^n for even n is real (+/- 1).
    # So we can integrate from 0 to inf and multiply by 2.
    
    def integrand_g(x):
        val = 0.0
        val += a0 * eval_hermite(0, x)
        val += a4 * eval_hermite(4, x)
        val += a8 * eval_hermite(8, x)
        val += a12 * eval_hermite(12, x)
        return np.abs(np.exp(-np.pi * x**2) * val)
        
    def integrand_fhat(xi):
        val = 0.0
        val += a0 * eval_hermite(0, xi) # (-i)^0 = 1
        val += a4 * (-1.0) * eval_hermite(4, xi) # (-i)^4 = 1
        val += a8 * 1.0 * eval_hermite(8, xi) # (-i)^8 = 1
        val += a12 * 1.0 * eval_hermite(12, xi) # (-i)^12 = 1
        # Wait, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
        # So Fhat(xi) = exp(-pi xi^2) * sum(ak * Hk(xi)) for even k.
        # Actually, (-i)^n = exp(-i pi n / 2).
        # n=0: 1
        # n=4: (-i)^4 = 1
        # n=8: (-i)^8 = 1
        # n=12: (-i)^12 = 1
        # So the Fourier transform has the same form as g, just with xi.
        return np.abs(np.exp(-np.pi * xi**2) * val)
        
    # Integration limits: Gaussian decays fast. 10 is safe.
    # Use quad for precision.
    try:
        norm_g, _ = quad(integrand_g, 0, 15, limit=100)
        norm_fhat, _ = quad(integrand_fhat, 0, 15, limit=100)
        
        if norm_fhat < 1e-12:
            return 1e12 # Large penalty
            
        ratio = (2 * norm_g) / (2 * norm_fhat)
        return ratio
    except Exception:
        return 1e12

def construct():
    start_time = time.time()
    deadline = start_time + 200.0 # Safety margin
    
    # Initial guess from seed
    seed_coeffs = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Define bounds for the search
    # The coefficients shouldn't be too large, otherwise the polynomial dominates and L1 norm blows up or oscillates wildly.
    # Seed is around -1, 0.04, 0.0003.
    # Let's search in a reasonable range.
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    best_score = objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Run Differential Evolution
    # Max iterations and population size tuned for time limit
    # DE is stochastic, so we rely on the deadline guard.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4,
            init='latinhypercube'
        )
        
        if result.success:
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
    except Exception as e:
        pass # Fallback to seed if DE fails or times out internally (though DE doesn't have internal timeout, we check wall clock)
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Local refinement using Nelder-Mead or L-BFGS-B on the best found
    from scipy.optimize import minimize
    
    # Try a few local starts around the best found and the seed
    local_starts = [best_coeffs, seed_coeffs]
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = best_coeffs + rng.normal(0, 0.1, 3)
        local_starts.append(pert)
        
    for start in local_starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs
