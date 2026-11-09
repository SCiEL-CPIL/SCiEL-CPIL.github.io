"""Model-and-optimize approach for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial basis with centered squared terms.
    
    This basis captures linear, quadratic, and pairwise interaction effects.
    Inputs are normalized to [-1, 1] for better conditioning.
    Squared terms are centered (subtracting 1/3) to be orthogonal to the constant term,
    which improves the conditioning of the design matrix for BFGS optimization.
    
    Parameters (10 total):
    0: constant
    1: u (normalized x)
    2: v (normalized y)
    3: w (normalized z)
    4: u^2 - 1/3 (centered quadratic x)
    5: v^2 - 1/3 (centered quadratic y)
    6: w^2 - 1/3 (centered quadratic z)
    7: u * v (interaction x,y)
    8: u * w (interaction x,z)
    9: v * w (interaction y,z)
    """
    p = params
    
    # Normalize inputs from [0.5, 2.0] to [-1, 1]
    # center = 1.25, half-range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Centered squared terms: E[u^2] = 1/3 for u ~ Uniform[-1, 1]
    # This reduces collinearity with the constant term.
    u2_c = u * u - 1.0 / 3.0
    v2_c = v * v - 1.0 / 3.0
    w2_c = w * w - 1.0 / 3.0
    
    # Interaction terms
    uv = u * v
    uw = u * w
    vw = v * w
    
    return (p[0] +
            p[1] * u +
            p[2] * v +
            p[3] * w +
            p[4] * u2_c +
            p[5] * v2_c +
            p[6] * w2_c +
            p[7] * uv +
            p[8] * uw +
            p[9] * vw)
# EVOLVE-BLOCK-END
