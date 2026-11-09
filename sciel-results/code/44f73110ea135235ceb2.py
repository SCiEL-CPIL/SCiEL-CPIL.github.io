"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 130.0  # Safety margin before the 150s hard kill
_START_TIME = time.time()


def _model(strain, temp, params):
    """
    Flexible Padé-like model for stress-strain with temperature effect.
    params length 10:
    p0: linear elastic modulus (E)
    p1: saturation stress (sigma_max)
    p2: rate parameter for plastic flow (affects curvature)
    p3: temp coefficient (linear shift)
    p4: temp-quadratic coefficient
    p5: offset
    p6: plastic exponent (affects shape of hardening)
    p7: temp-stress interaction
    p8: denominator shift (Padé)
    p9: plastic strain scale
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive scale for plastic strain to avoid singularities
    eps_scale = np.maximum(p9, 1e-4)
    
    # Strain-based hardening: saturates as strain increases
    # Use a form that captures linear elastic then plastic hardening
    elastic_term = p0 * strain
    # Plastic term: saturates via rational form
    # tanh-like behavior captured by x/(1+x/a) type
    plastic_strain = np.maximum(strain, 0.0)
    hardening = p1 * (plastic_strain / (plastic_strain + eps_scale)) ** p6
    
    # Temperature effects: thermal expansion + softening
    temp_effect = p3 * temp + p4 * temp**2
    # Interaction: temperature modifies the plastic flow
    temp_interaction = p7 * temp * plastic_strain
    
    # Combine with Padé-like denominator for flexibility
    numerator = elastic_term + hardening + temp_effect + temp_interaction + p5
    denominator = 1.0 + p8 * plastic_strain
    
    return numerator / denominator


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    return _model(strain, temp, params)
# EVOLVE-BLOCK-END
