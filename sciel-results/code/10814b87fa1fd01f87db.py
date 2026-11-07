"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration.
    
    Uses 10 parameters to define a sum of 3 Gaussian bumps in the (t, x, v) state space.
    Each bump has:
    - 1 weight (amplitude)
    - 3 centers (t_c, x_c, v_c)
    - 3 scales (sigma_t, sigma_x, sigma_v)
    Total: 3 * (1 + 3 + 3) = 21 parameters, but we only have 10.
    
    So we use a more efficient structure:
    - 3 weights
    - 3 shared center offsets (applied to t, x, v separately)
    - 3 shared scales
    - 1 global offset
    Total: 3 + 3 + 3 + 1 = 10 parameters
    
    Actually, let's be more flexible:
    - p0, p1, p2: weights for 3 Gaussian bumps
    - p3, p4, p5: t-centers for the 3 bumps
    - p6, p7, p8: x-centers for the 3 bumps
    - p9: v-center (shared across all bumps)
    
    Scales are fixed to reasonable values based on input ranges.
    """
    # Input ranges: t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    
    # Fixed scales based on input ranges
    sigma_t = 5.0   # ~1/4 of t range
    sigma_x = 0.1   # ~1/2 of x range
    sigma_v = 0.15  # ~1/2 of v range
    
    # Extract parameters
    w1, w2, w3 = params[0], params[1], params[2]
    tc1, tc2, tc3 = params[3], params[4], params[5]
    xc1, xc2, xc3 = params[6], params[7], params[8]
    vc = params[9]
    
    # Compute Gaussian bumps
    # Bump 1
    g1 = w1 * np.exp(-0.5 * ((t - tc1) / sigma_t)**2 - 0.5 * ((x - xc1) / sigma_x)**2 - 0.5 * ((v - vc) / sigma_v)**2)
    
    # Bump 2
    g2 = w2 * np.exp(-0.5 * ((t - tc2) / sigma_t)**2 - 0.5 * ((x - xc2) / sigma_x)**2 - 0.5 * ((v - vc) / sigma_v)**2)
    
    # Bump 3
    g3 = w3 * np.exp(-0.5 * ((t - tc3) / sigma_t)**2 - 0.5 * ((x - xc3) / sigma_x)**2 - 0.5 * ((v - vc) / sigma_v)**2)
    
    return g1 + g2 + g3
# EVOLVE-BLOCK-END
