"""Rewritten program using Local Linear RBF model for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Local Linear RBF model.
    
    The model is a sum of 3 RBF centers, each multiplying a local linear function
    (w0 + w1*x + w2*v), plus a global offset.
    
    params[0:3]:  Coeffs for RBF 1 (w0, w1, w2)
    params[3:6]:  Coeffs for RBF 2 (w0, w1, w2)
    params[6:9]:  Coeffs for RBF 3 (w0, w1, w2)
    params[9]:    Global offset
    """
    # Define RBF centers
    # Training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # Place centers to cover the domain effectively
    cx1, cv1 = -0.15, -0.28  # Bottom Left
    cx2, cv2 = 0.12, -0.28   # Bottom Right
    cx3, cv3 = 0.00, 0.25    # Top Middle
    
    # RBF bandwidth - wider for smoother interpolation and better coverage
    sigma = 0.22
    
    # Compute RBF values
    rbf1 = np.exp(-((x - cx1)**2 + (v - cv1)**2) / (2 * sigma**2))
    rbf2 = np.exp(-((x - cx2)**2 + (v - cv2)**2) / (2 * sigma**2))
    rbf3 = np.exp(-((x - cx3)**2 + (v - cv3)**2) / (2 * sigma**2))
    
    # Local linear terms
    local1 = params[0] + params[1] * x + params[2] * v
    local2 = params[3] + params[4] * x + params[5] * v
    local3 = params[6] + params[7] * x + params[8] * v
    
    # Global offset to capture mean acceleration
    offset = params[9]
    
    return rbf1 * local1 + rbf2 * local2 + rbf3 * local3 + offset
# EVOLVE-BLOCK-END
