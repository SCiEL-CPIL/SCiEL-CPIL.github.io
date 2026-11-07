"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Strategy: Use a fixed grid of Gaussian kernels centered in the normalized 
    (x, v) space. This provides a flexible, local basis that can approximate 
    arbitrary smooth nonlinear functions better than a global polynomial of 
    the same parameter count. The 10 parameters correspond to the weights of 
    9 Gaussian centers (3x3 grid) and 1 bias term.
    """
    p = params
    
    # Normalize inputs based on training ranges
    # x range: [-0.5728, 0.5275] -> center approx 0, scale approx 0.55
    # v range: [-0.1754, 0.2821] -> center approx 0, scale approx 0.25
    # We map inputs to a normalized space roughly in [-1.5, 1.5] to place centers comfortably
    x_norm = x / 0.55
    v_norm = v / 0.25
    
    # Define a 3x3 grid of centers in normalized space
    # Centers at -1, 0, 1 in both x and v normalized coords
    centers_x = np.array([-1.0, 0.0, 1.0])
    centers_v = np.array([-1.0, 0.0, 1.0])
    
    # Gaussian width (sigma) in normalized space
    # Needs to be wide enough to cover the domain but narrow enough to be local
    sigma = 0.8
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Initialize output
    out = np.zeros_like(x)
    
    # Iterate over the 9 centers
    # We map the 9 params (p[0] to p[8]) to the 9 centers
    # p[9] is the bias
    idx = 0
    for cx in centers_x:
        for cv in centers_v:
            # Calculate squared distance from each point to this center
            dx = x_norm - cx
            dv = v_norm - cv
            dist_sq = dx * dx + dv * dv
            
            # Gaussian kernel
            kernel = np.exp(-inv_2_sigma2 * dist_sq)
            
            # Add weighted contribution
            out += p[idx] * kernel
            idx += 1
            
    # Add bias
    out += p[9]
    
    return out
# EVOLVE-BLOCK-END
