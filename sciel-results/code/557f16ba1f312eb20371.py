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
        
    # Strategy: Nonlinear Gaussian Mixture Model (GMM)
    # We model y as a sum of 3 Gaussian "bumps" plus a linear trend.
    # This allows the model to capture localized features, peaks, and global trends.
    #
    # Parameters (10 total):
    # p[0], p[1], p[2]: Amplitudes (A1, A2, A3)
    # p[3], p[4], p[5]: Centers (mu1, mu2, mu3)
    # p[6], p[7], p[8]: Widths (sigma1, sigma2, sigma3) -> mapped via softplus to ensure > 0
    # p[9]:            Linear slope (m) for global trend
    
    # Extract parameters
    A1, A2, A3 = p[0], p[1], p[2]
    mu1, mu2, mu3 = p[3], p[4], p[5]
    
    # Map widths to positive values using softplus: log(1 + exp(x))
    # This is smooth and differentiable, unlike abs() or max(x, eps)
    # We scale down to keep widths in a reasonable range for [-3, 3]
    sigma1 = np.log1p(np.exp(p[6])) * 0.5
    sigma2 = np.log1p(np.exp(p[7])) * 0.5
    sigma3 = np.log1p(np.exp(p[8])) * 0.5
    
    m = p[9]
    
    # Compute Gaussian components
    # g(x) = A * exp(-0.5 * ((x - mu) / sigma)^2)
    
    # Vectorized computation
    x_expanded = x[:, np.newaxis]
    
    # Component 1
    diff1 = x_expanded - mu1
    g1 = A1 * np.exp(-0.5 * (diff1 / sigma1)**2)
    
    # Component 2
    diff2 = x_expanded - mu2
    g2 = A2 * np.exp(-0.5 * (diff2 / sigma2)**2)
    
    # Component 3
    diff3 = x_expanded - mu3
    g3 = A3 * np.exp(-0.5 * (diff3 / sigma3)**2)
    
    # Linear trend
    linear = m * x
    
    # Sum components
    result = g1 + g2 + g3 + linear
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback if numerical instability occurs (e.g., overflow in exp)
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
