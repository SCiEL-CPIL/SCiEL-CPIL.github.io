"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic Polynomial + 2 Gaussian Bumps.
    
    Params:
    p0, p1, p2, p3: Polynomial coefficients
    w1, c1, s1: Gaussian 1 (weight, center, std)
    w2, c2, s2: Gaussian 2 (weight, center, std)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    w1, c1, s1 = params[4], params[5], params[6]
    w2, c2, s2 = params[7], params[8], params[9]
    
    # Stabilize standard deviations to avoid division by zero or extreme values
    # Clip to [0.1, 10.0]
    s1_safe = np.clip(s1, 0.1, 10.0)
    s2_safe = np.clip(s2, 0.1, 10.0)
    
    # Polynomial component
    y_poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Gaussian components
    # Standard Gaussian: exp(-0.5 * z^2)
    z1 = (x - c1) / s1_safe
    z2 = (x - c2) / s2_safe
    
    g1 = w1 * np.exp(-0.5 * z1**2)
    g2 = w2 * np.exp(-0.5 * z2**2)
    
    # Handle potential overflow in exp (though z is bounded if x and c are reasonable)
    # x in [0.1, 6], c in R. If c is huge, z is huge, exp is 0. Safe.
    # If s is small (0.1), z can be large. exp(-large) -> 0. Safe.
    
    return y_poly + g1 + g2
# EVOLVE-BLOCK-END
