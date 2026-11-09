"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative interaction model:
    sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9) + p10
    
    This decouples:
    - Linear elastic scaling: (p0 + p1*strain)
    - Temperature-dependent stiffness: (p2 + p3*temp)
    - Non-linear strain saturation: (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9)
    - Offset: p10
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    # Ensure positive bases for power operations to avoid NaN
    strain_safe = np.clip(strain, -0.001, 1.0)
    temp_safe = np.clip(temp, 0.066, 1.0)
    
    # Linear elastic part
    elastic = p0 + p1 * strain_safe
    
    # Temperature modulation of stiffness
    temp_mod = p2 + p3 * temp_safe
    
    # Non-linear strain effects
    # Use np.power with safe exponents
    strain_pow5 = np.power(np.abs(strain_safe), p5) * np.sign(strain_safe)
    strain_pow7 = np.power(np.abs(strain_safe), p7) * np.sign(strain_safe)
    temp_pow9 = np.power(temp_safe, p9)
    
    numerator = 1.0 + p4 * strain_pow5
    denominator = 1.0 + p6 * strain_pow7 + p8 * temp_pow9
    
    # Avoid division by zero
    denom_safe = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    
    stress = elastic * temp_mod * (numerator / denom_safe) + p10
    
    return stress
# EVOLVE-BLOCK-END
