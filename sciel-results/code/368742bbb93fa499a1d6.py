"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Sum of 3 Gaussians with learnable centers, widths, and amplitudes.
    Symmetry breaking is applied to the centers to help BFGS converge from all-ones initialization.
    """
    p = params
    
    # Unpack parameters
    # p0: bias
    # p1, p2, p3: amp1, center1, log_width1
    # p4, p5, p6: amp2, center2, log_width2
    # p7, p8, p9: amp3, center3, log_width3
    
    bias = p[0]
    
    # Gaussian 1
    amp1 = p[1]
    center1 = p[2]
    log_w1 = np.clip(p[3], -10, 10) # Clip to prevent overflow/underflow
    width1 = np.exp(log_w1)
    gauss1 = amp1 * np.exp(-((x - center1) ** 2) / (2 * width1 ** 2))
    
    # Gaussian 2 (Symmetry breaking: +1 to center)
    amp2 = p[4]
    center2 = p[5] + 1.0
    log_w2 = np.clip(p[6], -10, 10)
    width2 = np.exp(log_w2)
    gauss2 = amp2 * np.exp(-((x - center2) ** 2) / (2 * width2 ** 2))
    
    # Gaussian 3 (Symmetry breaking: -1 to center)
    amp3 = p[7]
    center3 = p[8] - 1.0
    log_w3 = np.clip(p[9], -10, 10)
    width3 = np.exp(log_w3)
    gauss3 = amp3 * np.exp(-((x - center3) ** 2) / (2 * width3 ** 2))
    
    result = bias + gauss1 + gauss2 + gauss3
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
