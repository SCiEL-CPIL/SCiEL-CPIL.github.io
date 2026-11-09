"""Nonlinear oscillator symbolic regression with RBF-augmented local features."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: global oscillator features + local RBF corrections on (x, v)
    to capture non-additive nonlinear coupling in the phase plane.
    
    y = driving(t) + spring(x) + damping(v) + RBF(x,v) terms
    """
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Global features (4 params)
    x2 = x * x
    xv = x * v
    
    # Local RBF features on (x, v) plane - 6 RBF centers
    # Each RBF has one parameter, creating local patches
    # Centers spread across typical (x, v) ranges
    cx1, cv1 = -0.15, -0.10
    cx2, cv2 = 0.0, 0.0
    cx3, cv3 = 0.15, 0.10
    cx4, cv4 = -0.10, 0.15
    cx5, cv5 = 0.10, -0.15
    cx6, cv6 = 0.05, 0.05
    
    # Gaussian RBF with fixed width sigma=0.12
    sigma2 = 0.0288  # (0.12)^2
    
    rbf1 = np.exp(-((x - cx1)**2 + (v - cv1)**2) / sigma2)
    rbf2 = np.exp(-((x - cx2)**2 + (v - cv2)**2) / sigma2)
    rbf3 = np.exp(-((x - cx3)**2 + (v - cv3)**2) / sigma2)
    rbf4 = np.exp(-((x - cx4)**2 + (v - cv4)**2) / sigma2)
    rbf5 = np.exp(-((x - cx5)**2 + (v - cv5)**2) / sigma2)
    rbf6 = np.exp(-((x - cx6)**2 + (v - cv6)**2) / sigma2)
    
    y = (params[0] * sin_t
         + params[1] * cos_t
         + params[2] * x
         + params[3] * v
         + params[4] * rbf1
         + params[5] * rbf2
         + params[6] * rbf3
         + params[7] * rbf4
         + params[8] * rbf5
         + params[9] * rbf6)
    
    return y
# EVOLVE-BLOCK-END
