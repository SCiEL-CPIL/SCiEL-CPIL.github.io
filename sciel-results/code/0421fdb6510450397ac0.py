"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) Network.
    
    Structure:
    - 2 RBF centers in 2D space (cx1, cy1) and (cx2, cy2).
    - 2 Widths (sigma1, sigma2).
    - 2 Output weights (w1, w2).
    - 1 Global bias (b).
    
    Total parameters: 9 (using 9 of the 10 available).
    
    Model:
    f(x, y) = w1 * exp(-||[x,y] - [cx1, cy1]||^2 / (2 * sigma1^2)) 
             + w2 * exp(-||[x,y] - [cx2, cy2]||^2 / (2 * sigma2^2)) 
             + b
    
    Parameters (10 total):
    p0: cx1
    p1: cy1
    p2: sigma1
    p3: cx2
    p4: cy2
    p5: sigma2
    p6: w1
    p7: w2
    p8: b
    p9: unused (kept for compatibility)
    """
    if params is None or len(params) < 9:
        return np.zeros_like(x)
    
    # Extract parameters
    cx1, cy1, sig1 = params[0], params[1], params[2]
    cx2, cy2, sig2 = params[3], params[4], params[5]
    w1, w2, b = params[6], params[7], params[8]
    
    # Ensure sigmas are positive to avoid division by zero or negative variance in exponent
    # Using a small epsilon to keep gradients finite and stable
    eps = 1e-6
    s1 = np.maximum(np.abs(sig1), eps)
    s2 = np.maximum(np.abs(sig2), eps)
    
    # Compute squared distances
    # dist1_sq = (x - cx1)^2 + (y - cy1)^2
    dist1_sq = (x - cx1)**2 + (y - cy1)**2
    dist2_sq = (x - cx2)**2 + (y - cy2)**2
    
    # Compute RBF activations
    # exp(-dist_sq / (2 * sigma^2))
    # We use 2*sigma^2 in denominator as per standard Gaussian RBF
    rbf1 = np.exp(-dist1_sq / (2 * s1**2))
    rbf2 = np.exp(-dist2_sq / (2 * s2**2))
    
    # Linear combination
    return w1 * rbf1 + w2 * rbf2 + b
# EVOLVE-BLOCK-END
