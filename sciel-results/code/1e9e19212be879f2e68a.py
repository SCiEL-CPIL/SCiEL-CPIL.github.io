"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with Gaussian kernels for
    nonlinear oscillator acceleration prediction.
    
    Uses 10 parameters to define 2 RBF centers (cx1, cx2, cy1, cy2),
    2 widths (w1, w2), and 3 weights (m1, m2, m3) plus a bias.
    
    This local approximation can capture complex nonlinear dynamics
    that global polynomials miss.
    
    Inputs:
      x: position
      v: velocity
      params: array of 10 coefficients
      
    Returns:
      a: acceleration
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Extract parameters
    cx1 = p[0]
    cx2 = p[1]
    cy1 = p[2]
    cy2 = p[3]
    w1 = p[4]
    w2 = p[5]
    m1 = p[6]
    m2 = p[7]
    m3 = p[8]
    bias = p[9]
    
    # Ensure widths are positive to avoid numerical issues
    w1 = np.maximum(np.abs(w1), 1e-4)
    w2 = np.maximum(np.abs(w2), 1e-4)
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dy1 = v - cy1
    d1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = v - cy2
    d2_sq = dx2 * dx2 + dy2 * dy2
    
    # Gaussian kernels
    k1 = np.exp(-d1_sq / (2.0 * w1 * w1))
    k2 = np.exp(-d2_sq / (2.0 * w2 * w2))
    
    # Combine with weights
    a = m1 * k1 + m2 * k2 + m3 + bias
    
    return a
# EVOLVE-BLOCK-END
