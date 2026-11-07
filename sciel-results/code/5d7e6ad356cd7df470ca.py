# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: Linear-in-parameters RBF Regression
    # We use 5 fixed centers spread across the domain [-3, 3].
    # The 10 parameters are:
    # p[0..4]: Weights for the 5 RBFs
    # p[5]:    Linear coefficient (x)
    # p[6]:    Quadratic coefficient (x^2)
    # p[7]:    Cubic coefficient (x^3)
    # p[8]:    Quartic coefficient (x^4)
    # p[9]:    Bias term
    
    # Fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Fixed width (sigma) for RBFs
    # A width of 1.0 provides good local coverage with overlap for smoothness
    sigma = 1.0
    
    # Compute RBF features
    # x is shape (N,), centers is shape (5,)
    # diff is shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF: exp(-0.5 * (diff / sigma)^2)
    rbf = np.exp(-0.5 * (diff / sigma)**2)
    
    # Polynomial features
    # Include up to 4th order to allow for global trends that RBFs might miss
    # or to help with boundary behavior if RBFs decay too fast
    x2 = x**2
    x3 = x**3
    x4 = x**4
    
    # Weights
    w_rbf = p[0:5]
    w_lin = p[5]
    w_quad = p[6]
    w_cub = p[7]
    w_quart = p[8]
    bias = p[9]
    
    # Linear combination
    # rbf @ w_rbf gives shape (N,)
    result = rbf @ w_rbf + w_lin * x + w_quad * x2 + w_cub * x3 + w_quart * x4 + bias
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
