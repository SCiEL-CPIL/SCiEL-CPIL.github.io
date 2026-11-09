"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise model with separate compressive and tensile branches:
    - For strain >= 0: elastic + Voce hardening + power-law term
    - For strain < 0: elastic + compressive hardening
    Temperature affects all coefficients linearly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent base coefficients
    E = p0 + p1 * temp           # Elastic modulus
    Y_t = p2 + p3 * temp         # Tensile yield stress
    H_t = p4 + p5 * temp         # Tensile hardening rate
    Y_c = p6 + p7 * temp         # Compressive yield stress
    H_c = p8 + p9 * temp         # Compressive hardening rate
    
    # Elastic term (common to both regimes)
    elastic = E * strain
    
    # Tensile branch (strain >= 0)
    eps_t = np.maximum(strain, 0.0)
    H_t_eps = np.clip(H_t * eps_t, -50, 50)
    hardening_t = Y_t * (1.0 - np.exp(-H_t_eps))
    
    # Compressive branch (strain < 0)
    eps_c = np.minimum(strain, 0.0)
    H_c_eps = np.clip(H_c * (-eps_c), -50, 50)
    hardening_c = Y_c * (1.0 - np.exp(-H_c_eps))
    
    # Combine: use tensile hardening for positive strain, compressive for negative
    mask_t = strain >= 0.0
    hardening = np.where(mask_t, hardening_t, hardening_c)
    
    return elastic + hardening
# EVOLVE-BLOCK-END
