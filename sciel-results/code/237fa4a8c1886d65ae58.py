"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a Voce-type hardening model with temperature-dependent coefficients:
    stress = (E0 + E1*temp) * strain + (Y0 + Y1*temp) * (1 - exp(-(H0 + H1*temp) * max(strain, 0)))
    + (S0 + S1*temp) * strain^2 + (T0 + T1*temp) * strain^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp          # Elastic modulus (temperature-dependent)
    Y = p2 + p3 * temp          # Yield stress (temperature-dependent)
    H = p4 + p5 * temp          # Hardening rate (temperature-dependent)
    
    # Plastic strain (only for positive strain)
    eps_p = np.maximum(strain, 0.0)
    
    # Elastic term
    elastic = E * strain
    
    # Voce hardening term: saturates as strain increases
    # Clip H*eps_p to prevent overflow
    H_eps = np.clip(H * eps_p, -50, 50)
    hardening = Y * (1.0 - np.exp(-H_eps))
    
    # Higher-order correction terms
    quad = (p6 + p7 * temp) * strain**2
    cubic = (p8 + p9 * temp) * strain**3
    
    return elastic + hardening + quad + cubic
# EVOLVE-BLOCK-END
