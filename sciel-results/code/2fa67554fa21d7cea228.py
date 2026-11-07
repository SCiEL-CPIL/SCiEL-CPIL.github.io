"""Stress-strain model for an aluminium rod using a rational/linear-in-params basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a linear-in-parameters basis function approach for robust BFGS optimization.
    The basis functions capture:
    - Elastic linear region (strain)
    - Plastic hardening (quadratic/cubic strain terms)
    - Temperature dependence (linear temp, interactions)
    - Non-linear strain effects (higher order terms)
    
    This linear structure ensures the MSE loss is convex in params,
    allowing BFGS to reliably find the global minimum.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to improve numerical conditioning
    # Training ranges: strain [-0.001911, 0.9636], temp [0.06667, 1.0]
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Basis functions - linear in params
    # Elastic region
    b0 = 1.0
    b1 = s_norm
    b2 = t_norm
    # Plastic hardening
    b3 = s_norm ** 2
    b4 = s_norm ** 3
    # Temperature interactions
    b5 = s_norm * t_norm
    b6 = s_norm ** 2 * t_norm
    # Higher order strain
    b7 = s_norm ** 4
    # Temperature nonlinear
    b8 = t_norm ** 2
    # Interaction
    b9 = s_norm ** 3 * t_norm
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
