import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y.
    Uses a hybrid model: Linear terms + 2 Gaussian Radial Basis Functions (RBFs).
    
    Parameters (10 total):
    p[0]: Weight for RBF 1
    p[1]: Weight for RBF 2
    p[2]: Center x for RBF 1
    p[3]: Center y for RBF 1
    p[4]: Center x for RBF 2
    p[5]: Center y for RBF 2
    p[6]: Log of width (sigma) for RBFs. sigma = exp(p[6])
    p[7]: Bias term
    p[8]: Coefficient for linear term x
    p[9]: Coefficient for linear term y
    
    The model is:
    pred = b + a_x*x + a_y*y + w1*exp(-((x-c1x)^2 + (y-c1y)^2) / (2*sigma^2)) 
                   + w2*exp(-((x-c2x)^2 + (y-c2y)^2) / (2*sigma^2))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
        
    # Extract parameters
    w1 = p[0]
    w2 = p[1]
    c1x = p[2]
    c1y = p[3]
    c2x = p[4]
    c2y = p[5]
    log_sigma = p[6]
    bias = p[7]
    ax = p[8]
    ay = p[9]
    
    # Compute width safely
    # exp(log_sigma) can overflow if log_sigma is large, underflow if small.
    # We clamp log_sigma to a reasonable range to prevent numerical issues in exp.
    # exp(50) is ~5e21, exp(-50) is ~1e-22. 
    # If sigma is huge, the RBF becomes a constant. If sigma is tiny, it becomes a spike.
    # Clamping ensures stability.
    log_sigma_clamped = np.clip(log_sigma, -10.0, 10.0)
    sigma = np.exp(log_sigma_clamped)
    sigma_sq = sigma * sigma
    
    # Prevent division by zero if sigma is extremely small (though clipped, 2*sigma_sq could be very small)
    # If sigma_sq is 0, we treat the RBF as 0 (or 1 at center, but 0 is safer for gradients?)
    # Actually, if sigma is 0, the function is undefined. 
    # With clipping at -10, sigma ~ 4.5e-5, sigma_sq ~ 2e-9. This is fine for float64.
    
    # Compute RBF terms
    # dist1_sq = (x - c1x)^2 + (y - c1y)^2
    dx1 = x - c1x
    dy1 = y - c1y
    dist1_sq = dx1 * dx1 + dy1 * dy1
    
    # dist2_sq = (x - c2x)^2 + (y - c2y)^2
    dx2 = x - c2x
    dy2 = y - c2y
    dist2_sq = dx2 * dx2 + dy2 * dy2
    
    # Exponent terms
    # Clip the exponent to avoid underflow to 0 (which is fine) or overflow (if dist is negative? no, dist_sq >= 0)
    # The exponent is -dist_sq / (2 * sigma_sq). This is always <= 0.
    # So exp() will be in (0, 1]. No overflow risk. Underflow to 0 is numerically stable.
    exp1 = np.exp(-dist1_sq / (2.0 * sigma_sq))
    exp2 = np.exp(-dist2_sq / (2.0 * sigma_sq))
    
    # Combine
    pred = bias + ax * x + ay * y + w1 * exp1 + w2 * exp2
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
