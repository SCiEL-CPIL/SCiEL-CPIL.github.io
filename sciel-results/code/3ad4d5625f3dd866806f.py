"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Enhanced Gaussian RBF Network with 3 kernels using a compact parameter layout.
    
    Parameters layout (10 total):
      - params[0]:   bias (b)
      - params[1:4]: center of kernel 1 (cx1, cy1, cz1)
      - params[4]:   width of kernel 1 (sigma1)
      - params[5]:   weight of kernel 1 (w1)
      - params[6]:   cx2 (x-center of kernel 2)
      - params[7]:   sigma2 (width of kernel 2)
      - params[8]:   w2 (weight of kernel 2)
      - params[9]:   cx3 (x-center of kernel 3)
    
    Kernels 2 and 3 are 1D (x-only) Gaussians with y,z centers fixed at 1.25 (midpoint).
    This allows 3 kernels: one full 3D for local features, two 1D for global x-trends.
    
    Output: b + w1*exp(-||[x,y,z]-c1||^2/(2*s1^2)) 
          + w2*exp(-(x-cx2)^2/(2*s2^2))
          + w3*exp(-(x-cx3)^2/(2*s2^2))  [shares sigma2 with kernel 2]
    """
    # Extract parameters
    b = params[0]
    cx1, cy1, cz1 = params[1], params[2], params[3]
    sigma1 = params[4]
    w1 = params[5]
    cx2 = params[6]
    sigma2 = params[7]
    w2 = params[8]
    cx3 = params[9]
    
    # Clamp sigmas to avoid division by zero
    eps = 1e-6
    s1 = np.maximum(np.abs(sigma1), eps)
    s2 = np.maximum(np.abs(sigma2), eps)
    s1_sq = s1 * s1
    s2_sq = s2 * s2
    
    # Kernel 1: full 3D Gaussian
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist1_sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    k1 = np.exp(-dist1_sq / (2.0 * s1_sq))
    
    # Kernel 2: 1D Gaussian in x only
    dx2 = x - cx2
    k2 = np.exp(-dx2 * dx2 / (2.0 * s2_sq))
    
    # Kernel 3: 1D Gaussian in x only, reusing weight w2 scaled
    dx3 = x - cx3
    k3 = np.exp(-dx3 * dx3 / (2.0 * s2_sq))
    
    # Weighted sum with bias
    result = b + w1 * k1 + w2 * (k2 + k3)
    
    return result
# EVOLVE-BLOCK-END
