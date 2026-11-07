# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 8 fixed centers (corners of the domain) in 3D.
    10 params: [w1, w2, w3, w4, w5, w6, w7, w8, s, b]
    """
    p = params
    w = p[:8]
    s = p[8]
    b = p[9]
    
    # Ensure width is positive and smooth for BFGS
    s_safe = np.sqrt(s**2 + 1e-6)
    
    # Define fixed centers at the corners of [0.5, 2.0]^3
    # c1: (0.5, 0.5, 0.5)
    # c2: (2.0, 0.5, 0.5)
    # c3: (0.5, 2.0, 0.5)
    # c4: (0.5, 0.5, 2.0)
    # c5: (2.0, 2.0, 0.5)
    # c6: (2.0, 0.5, 2.0)
    # c7: (0.5, 2.0, 2.0)
    # c8: (2.0, 2.0, 2.0)
    
    # Calculate squared distances to each center
    d1_sq = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
    d2_sq = (x - 2.0)**2 + (y - 0.5)**2 + (z - 0.5)**2
    d3_sq = (x - 0.5)**2 + (y - 2.0)**2 + (z - 0.5)**2
    d4_sq = (x - 0.5)**2 + (y - 0.5)**2 + (z - 2.0)**2
    d5_sq = (x - 2.0)**2 + (y - 2.0)**2 + (z - 0.5)**2
    d6_sq = (x - 2.0)**2 + (y - 0.5)**2 + (z - 2.0)**2
    d7_sq = (x - 0.5)**2 + (y - 2.0)**2 + (z - 2.0)**2
    d8_sq = (x - 2.0)**2 + (y - 2.0)**2 + (z - 2.0)**2
    
    # Compute Gaussian kernels
    k1 = np.exp(-d1_sq / (2.0 * s_safe**2))
    k2 = np.exp(-d2_sq / (2.0 * s_safe**2))
    k3 = np.exp(-d3_sq / (2.0 * s_safe**2))
    k4 = np.exp(-d4_sq / (2.0 * s_safe**2))
    k5 = np.exp(-d5_sq / (2.0 * s_safe**2))
    k6 = np.exp(-d6_sq / (2.0 * s_safe**2))
    k7 = np.exp(-d7_sq / (2.0 * s_safe**2))
    k8 = np.exp(-d8_sq / (2.0 * s_safe**2))
    
    return w[0]*k1 + w[1]*k2 + w[2]*k3 + w[3]*k4 + w[4]*k5 + w[5]*k6 + w[6]*k7 + w[7]*k8 + b
# EVOLVE-BLOCK-END
