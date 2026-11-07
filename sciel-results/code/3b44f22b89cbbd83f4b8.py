"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a hyperbolic Ramberg-Osgood-type model with temperature-dependent
    elastic modulus and strain hardening terms.
    
    Model:
        E(T) = p0 + p1 * T
        stress = E(T) * strain / (1 + p2 * strain)
                 + p3 * strain^2 + p4 * strain^3
                 + p5 * T * strain + p6 * T^2 * strain
                 + p7 * T + p8 * T^2 + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * temp
    
    # Hyperbolic elastic-plastic transition
    elastic_plastic = E_T * strain / (1.0 + p2 * strain)
    
    # Strain hardening terms
    hardening = p3 * strain**2 + p4 * strain**3
    
    # Temperature-strain interaction
    temp_interaction = p5 * temp * strain + p6 * temp**2 * strain
    
    # Temperature-only terms
    temp_only = p7 * temp + p8 * temp**2
    
    stress = elastic_plastic + hardening + temp_interaction + temp_only + p9
    
    return stress
# EVOLVE-BLOCK-END
