import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    # Gaussian Mixture Model with 4 Gaussians
    # Each Gaussian: center (mu), width (sigma), amplitude (amp)
    # params: [mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3, mu4, sigma4, amp4]
    # But we only have 10 params, so let's use 3 Gaussians + 1 linear term
    # Actually, let's use 4 Gaussians but share some structure
    # Let's use: 3 Gaussians (mu, sigma, amp) = 9 params + 1 global offset = 10 params
    
    # Map x to [-1, 1] for better numerical stability
    t = x / 3.0
    
    # Extract parameters
    mu1 = params[0] * 3.0  # scale back to original domain
    sigma1 = np.abs(params[1]) * 3.0 + 0.1  # ensure positive width
    amp1 = params[2]
    
    mu2 = params[3] * 3.0
    sigma2 = np.abs(params[4]) * 3.0 + 0.1
    amp2 = params[5]
    
    mu3 = params[6] * 3.0
    sigma3 = np.abs(params[7]) * 3.0 + 0.1
    amp3 = params[8]
    
    offset = params[9]
    
    # Compute Gaussian terms
    # Use stable computation to avoid overflow/underflow
    g1 = amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    g3 = amp3 * np.exp(-0.5 * ((x - mu3) / sigma3) ** 2)
    
    y = offset + g1 + g2 + g3
    
    return y
