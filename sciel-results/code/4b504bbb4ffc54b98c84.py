"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h_init):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Metric: max_k |convolve(h, h)[k]| / (sum_i |h_i|)^2
    """
    
    def objective(x):
        # Map x to h in (-1, 1) using tanh to keep bounded
        h = np.tanh(x)
        
        # Compute convolution
        conv = np.convolve(h, h, mode='full')
        
        # Smooth absolute value and max
        eps = 1e-8
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Soft max with temperature T
        # Using a lower T makes it closer to true max but steeper gradients
        T = 2.0 
        max_abs = np.max(abs_conv)
        # Subtract max for numerical stability in exp
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        # Avoid division by zero
        denom = sum_abs_h**2 + eps
        value = log_max / denom
        return value

    # Use a random initialization to explore the landscape
    # The seed is all positive. A random mix might find a better local min
    # if the metric has multiple basins.
    rng = np.random.default_rng(42)
    x0 = rng.uniform(-5, 5, size=N)
    
    # Run minimization
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = np.tanh(res.x)
    
    # Compare with seed (all ones) to ensure we don't return something worse
    # Calculate seed value
    h_seed = np.ones(N)
    conv_seed = np.convolve(h_seed, h_seed, mode='full')
    val_seed = np.max(np.abs(conv_seed)) / (np.sum(np.abs(h_seed))**2)
    
    # Calculate optimized value (using exact formula for comparison, not soft max)
    conv_opt = np.convolve(h_opt, h_opt, mode='full')
    val_opt = np.max(np.abs(conv_opt)) / (np.sum(np.abs(h_opt))**2)
    
    # If optimization didn't improve, return seed
    if val_opt >= val_seed:
        return h_seed
    else:
        return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
