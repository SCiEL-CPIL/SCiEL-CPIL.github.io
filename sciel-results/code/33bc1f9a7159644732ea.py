"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
from scipy.special import gamma

def _compute_l2_norm_squared(roots):
    """
    Compute ||P||_{L^2(e^{-x})}^2 where P(x) = prod_{i=1}^{12} (x - r_i)^2.
    
    Uses polynomial expansion and the fact that int_0^inf x^k e^{-x} dx = k!.
    """
    roots = np.array(roots)
    n = len(roots)
    
    # Build P(x) = prod (x - r_i)^2
    # Start with 1
    coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2*r*x + r^2
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[:-2] += coeffs
        new_coeffs[1:-1] += -2 * r * coeffs
        new_coeffs[2:] += r**2 * coeffs
        coeffs = new_coeffs
    
    # Compute P(x)^2 coefficients
    n_coeff = len(coeffs)
    p2_coeffs = np.zeros(2 * n_coeff - 1)
    for i in range(n_coeff):
        if coeffs[i] == 0:
            continue
        for j in range(n_coeff):
            if coeffs[j] == 0:
                continue
            p2_coeffs[i + j] += coeffs[i] * coeffs[j]
    
    # Compute integral = sum_k p2_coeffs[k] * k!
    integral = 0.0
    for k in range(len(p2_coeffs)):
        if p2_coeffs[k] != 0:
            integral += p2_coeffs[k] * gamma(k + 1)
    
    return integral

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Optimizes the root placement to minimize the Cohn-Goncalves uncertainty bound
    by treating the 12 double-root locations as continuous variables.
    """
    # Get initial guess from Laguerre zeros
    zeros, _ = roots_laguerre(12)
    
    # Compute the bound at Laguerre zeros for normalization
    base_val = _compute_l2_norm_squared(zeros)
    base_sqrt = np.sqrt(base_val)
    
    # Define the objective function in log-space for positivity constraint
    def objective(params):
        roots = np.exp(params)
        try:
            val = _compute_l2_norm_squared(roots)
            if not np.isfinite(val) or val < 0:
                return 1e10
            return np.sqrt(val) / base_sqrt
        except Exception:
            return 1e10
    
    # Multi-start optimization
    best_val = np.inf
    best_params = None
    
    n_starts = 8
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if start == 0:
            x_init = np.log(zeros)
        elif start == 1:
            x_init = np.log(zeros) + 0.05 * rng.randn(12)
        elif start == 2:
            x_init = np.log(zeros) - 0.05 * rng.randn(12)
        elif start == 3:
            x_init = np.log(zeros) + 0.1 * rng.randn(12)
        elif start == 4:
            x_init = np.log(zeros) - 0.1 * rng.randn(12)
        elif start == 5:
            # Spread out more
            x_init = np.log(zeros) + 0.3 * rng.randn(12)
        elif start == 6:
            # Cluster more
            x_init = np.log(zeros) + 0.02 * rng.randn(12)
        else:
            # Random restart from different scale
            x_init = np.log(zeros) * rng.uniform(0.5, 1.5)
        
        try:
            result = minimize(objective, x_init, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            val = result.fun
            if val < best_val and np.isfinite(val):
                best_val = val
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        return [float(z) for z in zeros]
    
    optimal_roots = np.exp(best_params)
    optimal_roots = np.sort(optimal_roots)
    
    return [float(r) for r in optimal_roots]
# EVOLVE-BLOCK-END
