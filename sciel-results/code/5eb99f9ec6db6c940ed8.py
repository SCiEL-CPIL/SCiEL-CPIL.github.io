"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    Strategy: Gaussian Mixture Model (GMM) of 3 experts in (x, v) space.
    Params:
    p[0]: bias
    p[1]: x-center 1
    p[2]: x-center 2
    p[3]: x-center 3
    p[4]: v-center 1
    p[5]: v-center 2
    p[6]: v-center 3
    p[7]: amplitude 1
    p[8]: amplitude 2
    p[9]: amplitude 3
    """
    p = params
    
    # Extract parameters
    bias = p[0]
    
    # Centers
    cx1, cx2, cx3 = p[1], p[2], p[3]
    cv1, cv2, cv3 = p[4], p[5], p[6]
    
    # Amplitudes
    a1, a2, a3 = p[7], p[8], p[9]
    
    # Fixed width (inverse variance) to ensure stability and smoothness
    # Input ranges: x in [-0.34, 0.36], v in [-0.21, 0.17]
    # A width of 0.1 allows overlapping but distinct experts
    sigma2 = 0.01  # sigma = 0.1
    
    # Calculate squared distances for each expert
    # Expert 1
    dx1 = x - cx1
    dv1 = v - cv1
    dist2_1 = dx1 * dx1 + dv1 * dv1
    
    # Expert 2
    dx2 = x - cx2
    dv2 = v - cv2
    dist2_2 = dx2 * dx2 + dv2 * dv2
    
    # Expert 3
    dx3 = x - cx3
    dv3 = v - cv3
    dist2_3 = dx3 * dx3 + dv3 * dv3
    
    # Gaussian kernels
    # Using exp(-dist2 / (2 * sigma2))
    # sigma2 = 0.01, so 2*sigma2 = 0.02
    g1 = np.exp(-dist2_1 / 0.02)
    g2 = np.exp(-dist2_2 / 0.02)
    g3 = np.exp(-dist2_3 / 0.02)
    
    # Combine
    y = bias + a1 * g1 + a2 * g2 + a3 * g3
    
    return y

# EVOLVE-BLOCK-END
