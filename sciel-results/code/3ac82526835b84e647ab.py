"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a bivariate polynomial expansion of degree 3 (10 coefficients).
    This captures non-linear interactions between strain and temperature
    without assuming a specific physical constitutive law.
    """
    x = strain
    y = temp
    
    p = params
    
    # Coefficients:
    # p[0]: constant
    # p[1]: x
    # p[2]: y
    # p[3]: x^2
    # p[4]: x*y
    # p[5]: y^2
    # p[6]: x^3
    # p[7]: x^2*y
    # p[8]: x*y^2
    # p[9]: y^3
    
    return (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * x**2 + 
            p[4] * x * y + 
            p[5] * y**2 + 
            p[6] * x**3 + 
            p[7] * x**2 * y + 
            p[8] * x * y**2 + 
            p[9] * y**3)
# EVOLVE-BLOCK-END
