"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a rational function.
    
    y = N(x, v) / D(x, v)
    
    N(x, v) = p0 + p1*x + p2*x^2 + p3*v + p4*x*v
    D(x, v) = 1 + p5*x + p6*x^2 + p7*v + p8*x*v + p9*v^2
    
    This structure allows for more flexible modeling of nonlinear dependencies
    compared to a pure polynomial, potentially capturing asymptotic behaviors
    or interactions that polynomials of the same order miss.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute numerator
    numerator = p0 + p1 * x + p2 * (x * x) + p3 * v + p4 * (x * v)
    
    # Compute denominator
    # Add a small epsilon to prevent division by zero if the denominator
    # approaches zero during optimization, though starting from 1s it should be safe.
    denominator = 1.0 + p5 * x + p6 * (x * x) + p7 * v + p8 * (x * v) + p9 * (v * v)
    
    # Ensure denominator is not too close to zero to avoid numerical issues
    # We can clip the absolute value of the denominator to be at least 1e-6
    # However, simply dividing is usually fine if the optimizer stays in a reasonable region.
    # To be safe for BFGS which might take large steps:
    denom_safe = np.where(np.abs(denominator) < 1e-6, np.sign(denominator) * 1e-6, denominator)
    # If denominator is exactly 0, sign is 0, so we need a fallback
    denom_safe = np.where(denom_safe == 0, 1e-6, denom_safe)
    
    result = numerator / denom_safe
    
    return result
# EVOLVE-BLOCK-END
