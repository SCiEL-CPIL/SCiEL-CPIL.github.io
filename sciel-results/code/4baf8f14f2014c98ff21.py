"""Improved symbolic regression for aluminium stress-strain with temperature."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a pure bivariate polynomial chaos expansion (Taylor series) up to degree 4.
    This removes structural bias of physical constitutive laws and provides a smooth,
    differentiable surface for BFGS optimization.
    
    The expansion uses normalized strain and temperature to improve conditioning:
    s = (strain - mean_s) / std_s
    t = (temp - mean_t) / std_t
    
    With 10 parameters, we use a truncated polynomial basis.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Normalize inputs for better conditioning
    # Training ranges: strain in [-0.001911, 0.9636], temp in [0.06667, 1]
    s = (strain - 0.48) / 0.5
    t = (temp - 0.53) / 0.47
    
    # Build polynomial basis terms (up to degree 4, truncated to 10 terms)
    # Term 0: constant
    # Term 1: s
    # Term 2: s^2
    # Term 3: s^3
    # Term 4: s^4
    # Term 5: t
    # Term 6: t^2
    # Term 7: s*t
    # Term 8: s^2*t
    # Term 9: s*t^2
    
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    t2 = t * t
    
    # Polynomial expansion
    stress = (
        params[0]
        + params[1] * s
        + params[2] * s2
        + params[3] * s3
        + params[4] * s4
        + params[5] * t
        + params[6] * t2
        + params[7] * s * t
        + params[8] * s2 * t
        + params[9] * s * t2
    )
    
    return stress
# EVOLVE-BLOCK-END
