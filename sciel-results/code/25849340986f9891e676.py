"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
      E(T) = p0 + p1 * temp
      stress = E(T) * strain / (1 + p2 * strain) 
               + p3 * strain^2 + p4 * strain^3
               + p5 * temp * strain + p6 * temp^2 * strain
               + p7 * temp + p8 * temp^2 + p9
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Elastic-plastic transition via rational form
    elastic_plastic = E * strain / (1.0 + p2 * strain)
    
    # Strain hardening terms
    hardening = p3 * strain * strain + p4 * strain * strain * strain
    
    # Temperature-strain interaction terms
    temp_interaction = p5 * temp * strain + p6 * temp * temp * strain
    
    # Pure temperature terms
    temp_only = p7 * temp + p8 * temp * temp + p9
    
    stress = elastic_plastic + hardening + temp_interaction + temp_only
    
    return stress
# EVOLVE-BLOCK-END
