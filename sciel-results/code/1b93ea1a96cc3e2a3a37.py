"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a polynomial expansion in strain with linear temperature dependence,
    augmented by a bounded hyperbolic tangent term to capture non-linear
    features like yield points or strain hardening knees.
    
    sigma = (p0 + p1*T)*eps + (p2 + p3*T)*eps^2 + (p4 + p5*T)*eps^3 + (p6 + p7*T)*tanh(p8*eps) + p9
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms with temperature dependence
    # Term 1: Linear in eps, linear in T
    term1 = (p0 + p1 * T) * eps
    
    # Term 2: Quadratic in eps, linear in T
    term2 = (p2 + p3 * T) * eps**2
    
    # Term 3: Cubic in eps, linear in T
    term3 = (p4 + p5 * T) * eps**3
    
    # Term 4: Non-linear bounded term with temperature dependence
    # tanh helps capture the "knee" in the stress-strain curve
    term4 = (p6 + p7 * T) * np.tanh(p8 * eps)
    
    # Constant offset
    term5 = p9
    
    return term1 + term2 + term3 + term4 + term5
# EVOLVE-BLOCK-END
