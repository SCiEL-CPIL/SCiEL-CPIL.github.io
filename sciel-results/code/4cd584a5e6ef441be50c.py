"""Symbolic regression using Gaussian Mixture Model approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model approximation:
    y = sum(A_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2))) + c
    
    where sigma_i = exp(theta_i) to ensure positivity and smoothness.
    
    params layout (length 10):
      params[0] = A1 (amplitude 1)
      params[1] = mu1 (center 1)
      params[2] = theta1 (log-width 1, sigma1 = exp(theta1))
      params[3] = A2 (amplitude 2)
      params[4] = mu2 (center 2)
      params[5] = theta2 (log-width 2, sigma2 = exp(theta2))
      params[6] = A3 (amplitude 3)
      params[7] = mu3 (center 3)
      params[8] = theta3 (log-width 3, sigma3 = exp(theta3))
      params[9] = c (bias)
    
    Parameters:
    x: Input array, shape (N,)
    params: Coefficients array, shape (10,)
    
    Returns:
    Predicted y values, shape (N,)
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Extract parameters
    A1, mu1, theta1 = params[0], params[1], params[2]
    A2, mu2, theta2 = params[3], params[4], params[5]
    A3, mu3, theta3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute sigmas using exp to ensure positivity
    # Clip theta to prevent overflow/underflow issues during optimization
    # exp(10) is huge, exp(-10) is tiny. 
    # Typical sigma for range [-2,2] is around 0.5 to 2.0. 
    # log(0.5) ~ -0.69, log(2) ~ 0.69.
    # Clipping theta to [-5, 5] is safe.
    theta1 = np.clip(theta1, -5.0, 5.0)
    theta2 = np.clip(theta2, -5.0, 5.0)
    theta3 = np.clip(theta3, -5.0, 5.0)
    
    sigma1 = np.exp(theta1)
    sigma2 = np.exp(theta2)
    sigma3 = np.exp(theta3)
    
    # Compute Gaussian terms
    # Use vectorized operations for speed
    g1 = A1 * np.exp(-((x - mu1) ** 2) / (2.0 * sigma1 ** 2))
    g2 = A2 * np.exp(-((x - mu2) ** 2) / (2.0 * sigma2 ** 2))
    g3 = A3 * np.exp(-((x - mu3) ** 2) / (2.0 * sigma3 ** 2))
    
    result = g1 + g2 + g3 + c
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
