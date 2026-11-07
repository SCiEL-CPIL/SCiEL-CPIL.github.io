"""
Strategy: Hybrid RBF-Polynomial Model with Learnable Width.
The model is:
f(x) = sum_{i=1}^{5} w_i * exp(-(x - c_i)^2 / (2 * sigma^2)) + p_0 + p_1*x + p_2*x^2 + p_3*x^3

Parameters (length 10):
params[0:5] : Weights w_1 to w_5
params[5]   : Log-width sigma (actual sigma = exp(params[5]))
params[6:10]: Polynomial coefficients p_0 to p_3

Centers c_i are fixed at -2.0, -1.0, 0.0, 1.0, 2.0.

This approach is nonlinear in params[5] (sigma), allowing the model to adapt
the spatial frequency of the local features. BFGS can optimize this.
The polynomial part handles the global trend.
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        # Fallback if params are malformed, though evaluator should provide 10
        return np.zeros_like(x)
        
    w = params[0:5]
    log_sigma = params[5]
    p_coeffs = params[6:10]
    
    # Compute sigma, ensuring it's positive and within reasonable bounds to avoid overflow/underflow
    # exp(log_sigma) can overflow if log_sigma is large. Clip it.
    # If log_sigma > 20, sigma is huge (~4e8), RBFs are flat.
    # If log_sigma < -20, sigma is tiny (~2e-9), RBFs are spikes.
    # Clipping to [-20, 20] is safe for exp.
    log_sigma_clipped = np.clip(log_sigma, -20.0, 20.0)
    sigma = np.exp(log_sigma_clipped)
    sigma_sq = sigma * sigma
    
    # Fixed centers
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Compute RBF terms
    # x is (N,), centers is (5,)
    # We want (N, 5) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-(diff ** 2) / (2.0 * sigma_sq))
    
    # Weighted sum of RBFs
    rbf_part = rbf_terms @ w
    
    # Compute Polynomial part
    # p_0 + p_1*x + p_2*x^2 + p_3*x^3
    poly_part = (p_coeffs[0] + 
                 p_coeffs[1] * x + 
                 p_coeffs[2] * x ** 2 + 
                 p_coeffs[3] * x ** 3)
    
    result = rbf_part + poly_part
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
