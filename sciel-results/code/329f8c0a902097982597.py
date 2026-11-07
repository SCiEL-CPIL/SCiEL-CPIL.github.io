"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis in (x, v) phase space with Fourier time modulation.
    
    Uses 5 RBF centers in the (x, v) plane combined with sin/cos(t) terms,
    plus a bias. This provides localized nonlinear basis functions that can
    capture complex oscillator dynamics without global polynomial ill-conditioning.
    
    Inputs are scaled to O(1) for better BFGS conditioning.
    """
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Scale inputs to roughly [-1, 1] range
    # x range: [-0.34, 0.36] -> scale ~ 0.35
    # v range: [-0.21, 0.17] -> scale ~ 0.20
    scale_x = 0.35
    scale_v = 0.20
    
    xs = x / scale_x
    vs = v / scale_v
    
    # RBF centers in (xs, vs) space - spread across the domain
    # Center 1: origin
    c1_x, c1_v = 0.0, 0.0
    # Center 2: positive x, slight negative v
    c2_x, c2_v = 0.5, -0.3
    # Center 3: negative x, positive v
    c3_x, c3_v = -0.5, 0.3
    # Center 4: positive x, positive v
    c4_x, c4_v = 0.3, 0.5
    # Center 5: negative x, negative v
    c5_x, c5_v = -0.3, -0.5
    
    # RBF width - moderate to cover the domain
    sigma = 0.8
    
    # Compute squared distances from each center
    d1 = (xs - c1_x)**2 + (vs - c1_v)**2
    d2 = (xs - c2_x)**2 + (vs - c2_v)**2
    d3 = (xs - c3_x)**2 + (vs - c3_v)**2
    d4 = (xs - c4_x)**2 + (vs - c4_v)**2
    d5 = (xs - c5_x)**2 + (vs - c5_v)**2
    
    # Gaussian RBFs
    rbf1 = np.exp(-d1 / (2.0 * sigma**2))
    rbf2 = np.exp(-d2 / (2.0 * sigma**2))
    rbf3 = np.exp(-d3 / (2.0 * sigma**2))
    rbf4 = np.exp(-d4 / (2.0 * sigma**2))
    rbf5 = np.exp(-d5 / (2.0 * sigma**2))
    
    # Fourier time terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Basis terms:
    # p0: bias
    # p1-p5: RBF coefficients
    # p6: RBF1 * sin(t)
    # p7: RBF1 * cos(t)
    # p8: RBF2 * sin(t)
    # p9: RBF2 * cos(t)
    
    result = (p0 + 
              p1 * rbf1 + 
              p2 * rbf2 + 
              p3 * rbf3 + 
              p4 * rbf4 + 
              p5 * rbf5 +
              p6 * rbf1 * sin_t +
              p7 * rbf1 * cos_t +
              p8 * rbf2 * sin_t +
              p9 * rbf2 * cos_t)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
