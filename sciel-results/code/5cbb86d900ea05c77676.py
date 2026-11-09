"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Rational Function Approximation
    # This is a distinct approach from the previous polynomial basis.
    # Basis:
    # Numerator: 1 + p0*x + p1*y + p2*z + p3*xy + p4*xz + p5*yz
    # Denominator: 1 + p6*x + p7*y + p8*z + p9*xy
    
    p = params
    
    # Calculate numerator
    num = (1.0 
           + p[0] * x 
           + p[1] * y 
           + p[2] * z 
           + p[3] * x * y 
           + p[4] * x * z 
           + p[5] * y * z)
           
    # Calculate denominator
    den = (1.0 
           + p[6] * x 
           + p[7] * y 
           + p[8] * z 
           + p[9] * x * y)
           
    # Ensure numerical stability: avoid division by zero or very small numbers
    # Since inputs are in [0.5, 2], and we start with positive params, 
    # the denominator is likely positive. We clip the absolute value to be >= 1e-6.
    # If the sign is negative, we keep the sign but ensure magnitude is >= 1e-6.
    sign = np.sign(den)
    sign[sign == 0] = 1.0
    den_magnitude = np.maximum(np.abs(den), 1e-6)
    den_safe = sign * den_magnitude
    
    return num / den_safe
# EVOLVE-BLOCK-END
