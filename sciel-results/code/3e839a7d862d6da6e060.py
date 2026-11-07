"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Pure polynomial chaos expansion (bivariate Taylor series) in strain
    and temperature with up to 10 coefficients. This provides a smooth,
    differentiable surface that captures non-linear interactions without
    the structural bias of physical constitutive laws.
    
    We use a 9-term polynomial (up to 10 params):
    stress = c0 + c1*eps + c2*T + c3*eps^2 + c4*eps*T + c5*T^2
             + c6*eps^3 + c7*eps^2*T + c8*eps*T^2
    
    Parameters p0..p9 correspond to c0..c9 (but we only have 10 params total).
    Actually, let's use 10 coefficients for a 4th order mixed polynomial.
    
    Let's define:
    p0: constant
    p1: strain
    p2: temp
    p3: strain^2
    p4: strain*temp
    p5: temp^2
    p6: strain^3
    p7: strain^2*temp
    p8: strain*temp^2
    p9: temp^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Bivariate polynomial expansion
    stress = (p0 
              + p1 * s 
              + p2 * t 
              + p3 * s**2 
              + p4 * s * t 
              + p5 * t**2 
              + p6 * s**3 
              + p7 * s**2 * t 
              + p8 * s * t**2 
              + p9 * t**3)
    
    return stress
# EVOLVE-BLOCK-END
