import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-style Radial Basis Function (RBF) model.
    
    Strategy:
    - Uses 5 fixed centers evenly spaced across the domain [-3, 3].
    - The model is y(x) = sum_{i=1}^5 w_i * exp(-0.5 * ((x - c_i) / sigma_i)^2)
    - Parameters 0-4 are the weights w_i.
    - Parameters 5-9 are the inverse widths (or scales) for the Gaussian kernels.
      We use sigma_i = 1.0 / (0.1 + abs(params[5+i])) to ensure positive, non-zero widths.
      This parameterization helps BFGS handle the scale better than raw sigma.
    
    This is a non-linear model in parameters (due to the width modulation), 
    but the landscape is generally smoother than pure polynomials for local features.
    """
    t_start = time.time()
    
    # Ensure x is a 1D array of floats
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers evenly spaced in [-3, 3]
    # Centers: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Weights
    w = params[0:5]
    
    # Widths (sigma)
    # We map params[5:10] to positive sigma values.
    # Using a softplus-like or simple absolute value with offset to avoid zero division.
    # sigma = 1 / (0.1 + |param|) gives sigma in (0, 10]. 
    # If param is large, sigma is small (sharp peak). If param is small, sigma is large (broad).
    # Let's try sigma = 1.0 / (0.1 + np.abs(params[5:10]))
    inv_sigmas = 0.1 + np.abs(params[5:10])
    sigmas = 1.0 / inv_sigmas
    
    # Check deadline early to avoid heavy computation if we are running late
    if time.time() - t_start > 10:
         # Fallback to a simple linear model if we are extremely slow (should not happen)
         return params[0] + params[1] * x

    # Compute RBFs
    # x shape: (N,)
    # centers shape: (5,)
    # We want (N, 5) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared distances
    diff_sq = diff * diff
    
    # Gaussian RBFs: exp(-0.5 * (diff / sigma)^2)
    # sigmas shape: (5,) -> broadcast to (1, 5)
    sigmas_sq = sigmas * sigmas
    scaled_sq = diff_sq / sigmas_sq[np.newaxis, :]
    
    # Compute exponentials
    rbf_values = np.exp(-0.5 * scaled_sq)
    
    # Check deadline again
    if time.time() - t_start > 120:
        return np.zeros_like(x)
        
    # Weighted sum
    # w shape: (5,) -> broadcast to (1, 5)
    y = np.sum(rbf_values * w[np.newaxis, :], axis=1)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
