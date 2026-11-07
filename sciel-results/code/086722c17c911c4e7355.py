import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a Gaussian Process-inspired kernel basis.
    
    Model:
    a = p0*X + p1*V 
      + p2 * exp(-0.5 * ((X-p3)^2 + (V-p4)^2) / (p5^2 + eps))
      + p6 * exp(-0.5 * ((X-p7)^2 + (V-p8)^2) / (p9^2 + eps))
      
    Inputs are scaled to O(1) range.
    p0, p1: Linear coefficients for scaled x and v.
    p2-p5: Amplitude, center_x, center_v, scale for first Gaussian bump.
    p6-p9: Amplitude, center_x, center_v, scale for second Gaussian bump.
    """
    # Scale inputs to O(1)
    X = x / 0.2
    V = v / 0.3
    
    # Extract parameters
    w0 = params[0]
    w1 = params[1]
    
    amp1 = params[2]
    cx1 = params[3]
    cv1 = params[4]
    s1_sq = params[5]**2 + 1e-6
    
    amp2 = params[6]
    cx2 = params[7]
    cv2 = params[8]
    s2_sq = params[9]**2 + 1e-6
    
    # Compute Gaussian terms
    diff_x1 = X - cx1
    diff_v1 = V - cv1
    r1_sq = diff_x1**2 + diff_v1**2
    g1 = np.exp(-0.5 * r1_sq / s1_sq)
    
    diff_x2 = X - cx2
    diff_v2 = V - cv2
    r2_sq = diff_x2**2 + diff_v2**2
    g2 = np.exp(-0.5 * r2_sq / s2_sq)
    
    # Combine
    pred = w0 * X + w1 * V + amp1 * g1 + amp2 * g2
    
    return pred

# EVOLVE-BLOCK-END
