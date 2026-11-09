import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic polynomial + 2 adaptive Gaussian bumps.
    params: [p0, p1, p2, p3, mu1, log_sigma1, amp1, mu2, log_sigma2, amp2]
    """
    x = np.asarray(x, dtype=float).ravel()
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    mu1 = params[4]
    log_sigma1 = params[5]
    amp1 = params[6]
    
    mu2 = params[7]
    log_sigma2 = params[8]
    amp2 = params[9]
    
    # Polynomial part
    # Using Horner's method for stability and speed
    poly = p0 + x * (p1 + x * (p2 + x * p3))
    
    # Gaussian 1
    # Clip log_sigma to keep sigma in [exp(-3), exp(2.3)] approx [0.05, 10]
    log_sigma1_clipped = np.clip(log_sigma1, -3.0, 2.3)
    sigma1 = np.exp(log_sigma1_clipped)
    # Avoid division by zero if sigma is extremely small, though clipping prevents this
    z1 = (x - mu1) / sigma1
    gauss1 = amp1 * np.exp(-0.5 * z1**2)
    
    # Gaussian 2
    log_sigma2_clipped = np.clip(log_sigma2, -3.0, 2.3)
    sigma2 = np.exp(log_sigma2_clipped)
    z2 = (x - mu2) / sigma2
    gauss2 = amp2 * np.exp(-0.5 * z2**2)
    
    pred = poly + gauss1 + gauss2
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
    return pred
