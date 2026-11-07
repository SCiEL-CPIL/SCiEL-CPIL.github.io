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
        # Pad with zeros
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Parameter mapping for 3 Gaussian Bumps
    # p[0], p[1], p[2] -> Centers (mu)
    # p[3], p[4], p[5] -> Widths (sigma)
    # p[6], p[7], p[8] -> Heights (A)
    # p[9] -> Global Offset (bias)
    
    mu1, mu2, mu3 = p[0], p[1], p[2]
    # Ensure sigma is positive and not too small to avoid division by zero/overflow
    # Using softplus-like behavior or just clipping
    sigma1 = np.clip(p[3], 0.1, 10.0)
    sigma2 = np.clip(p[4], 0.1, 10.0)
    sigma3 = np.clip(p[5], 0.1, 10.0)
    
    A1, A2, A3 = p[6], p[7], p[8]
    offset = p[9]
    
    # Calculate Gaussians
    # exp(-0.5 * ((x - mu) / sigma)^2)
    # To prevent overflow in exp, we clip the argument
    z1 = -0.5 * ((x - mu1) / sigma1) ** 2
    z2 = -0.5 * ((x - mu2) / sigma2) ** 2
    z3 = -0.5 * ((x - mu3) / sigma3) ** 2
    
    # Clip z to avoid underflow issues (though underflow to 0 is fine, overflow is not)
    # z is always <= 0, so exp(z) <= 1. No overflow risk from positive exponent.
    # However, if sigma is tiny, (x-mu)/sigma can be huge, squared is huge, z is -huge.
    # exp(-huge) is 0. This is numerically stable.
    
    g1 = np.exp(z1)
    g2 = np.exp(z2)
    g3 = np.exp(z3)
    
    result = A1 * g1 + A2 * g2 + A3 * g3 + offset
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return params[0] * x + params[1]
    
    return result
# EVOLVE-BLOCK-END
