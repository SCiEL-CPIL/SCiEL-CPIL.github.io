"""Rational function (Padé-like) model for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a ratio of two quadratic polynomials.
    
    Inputs are normalized to [0, 1] based on known training ranges:
    - b (population density): [1, 4.998]
    - s (substrate): [1, 4.999]
    - temp: [1.002, 39.98]
    - pH: [2, 12]
    
    The model is:
      y = (p0 + p1*b + p2*s + p3*temp + p4*pH + p5*b*s + p6*b*temp + p7*s*temp + p8*pH*b + p9*pH*s)
          / (1 + p10*b + p11*s + p12*temp + p13*pH + p14*b*s + p15*b*temp + p16*s*temp + p17*pH*b + p18*pH*s)
    
    But we only have 10 params. So let's use a simpler rational form:
      y = (a0 + a1*b + a2*s + a3*temp + a4*pH) / (1 + a5*b + a6*s + a7*temp + a8*pH + a9*b*s)
    
    This gives us 10 parameters total.
    """
    # Unpack parameters
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = params
    
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    pH_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Numerator: linear combination of normalized inputs
    numerator = (a0 + a1 * b_norm + a2 * s_norm + a3 * temp_norm + a4 * pH_norm)
    
    # Denominator: 1 + linear terms + one interaction term
    denominator = (1.0 + a5 * b_norm + a6 * s_norm + a7 * temp_norm + a8 * pH_norm + a9 * b_norm * s_norm)
    
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    
    return numerator / denominator
# EVOLVE-BLOCK-END
