# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Bivariate polynomial chaos expansion (total degree 3)
    stress = sum of 10 monomials: 1, s, t, s^2, st, t^2, s^3, s^2*t, s*t^2, t^3
    
    This is a smooth, differentiable surface with no structural bias from
    physical constitutive laws. All 10 parameters are linear coefficients,
    making the optimization problem well-behaved for BFGS.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    s = strain
    t = temp
    
    # Build all 10 monomials up to total degree 3
    stress = (c0 * np.ones_like(s) +
              c1 * s +
              c2 * t +
              c3 * s**2 +
              c4 * s * t +
              c5 * t**2 +
              c6 * s**3 +
              c7 * s**2 * t +
              c8 * s * t**2 +
              c9 * t**3)
    
    return stress
# EVOLVE-BLOCK-END
