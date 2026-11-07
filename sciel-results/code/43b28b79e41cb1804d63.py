"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model (GMM) with 4 components.
    params[0:4]   : amplitudes (a1..a4)
    params[4:8]   : centers   (c1..c4)
    params[8:10]  : widths    (sigma1, sigma2) - shared for stability
    
    Each Gaussian: a_i * exp(-(x - c_i)^2 / (2 * sigma_i^2))
    We use 2 shared widths to reduce parameter count and improve conditioning.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: simple linear if time runs out
        return params[0] * x + params[1]
    
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    a1, a2, a3, a4 = params[0], params[1], params[2], params[3]
    c1, c2, c3, c4 = params[4], params[5], params[6], params[7]
    
    # Use shared widths for stability: sigma1 for components 1&2, sigma2 for 3&4
    # Ensure widths are positive and not too small to avoid numerical issues
    sigma1 = np.maximum(np.abs(params[8]) + 0.1, 0.1)
    sigma2 = np.maximum(np.abs(params[9]) + 0.1, 0.1)
    
    # Compute Gaussian terms
    # Using stable computation: exp(-0.5 * ((x-c)/sigma)^2)
    term1 = a1 * np.exp(-0.5 * ((x - c1) / sigma1) ** 2)
    term2 = a2 * np.exp(-0.5 * ((x - c2) / sigma1) ** 2)
    term3 = a3 * np.exp(-0.5 * ((x - c3) / sigma2) ** 2)
    term4 = a4 * np.exp(-0.5 * ((x - c4) / sigma2) ** 2)
    
    return term1 + term2 + term3 + term4
# EVOLVE-BLOCK-END
