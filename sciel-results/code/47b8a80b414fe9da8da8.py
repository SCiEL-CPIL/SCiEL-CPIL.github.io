"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a rational function structure to capture non-monotonic stress-strain
    behavior and temperature effects. The model combines:
    - A linear elastic term
    - A nonlinear hardening term with saturation
    - Temperature-dependent yield stress
    - Interaction terms
    
    params: array of length 10
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize temperature to [0, 1] range for numerical stability
    # Training range: [0.06667, 1]
    t_norm = (temp - 0.06667) / (1.0 - 0.06667)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Handle negative strain (compression) - stress should be negative
    # Use sign-preserving transformations
    
    # Core strain term with smooth nonlinearity
    # Use a rational function that captures elastic-plastic transition
    strain_abs = np.abs(strain)
    strain_sign = np.sign(strain)
    
    # Elastic-plastic transition using smooth rational function
    # This creates a curve that is linear at small strain and saturates at large strain
    eps = 1e-8
    denom = 1.0 + p4 * strain_abs
    nonlinear_strain = strain_abs / denom
    
    # Hardening term with temperature dependence
    # Yield stress decreases with temperature
    yield_temp_factor = 1.0 - p3 * t_norm
    
    # Combined stress model
    # Linear elastic component
    elastic = p0 * strain
    
    # Plastic hardening component (only for positive strain, symmetric for compression)
    hardening = p1 * nonlinear_strain * strain_sign * yield_temp_factor
    
    # Temperature effect on baseline stress
    temp_effect = p2 * t_norm
    
    # Interaction between strain and temperature
    interaction = p5 * strain * t_norm
    
    # Cubic correction for higher-order effects
    cubic_correction = p6 * strain**3
    
    # Saturation limit term - prevents unbounded growth
    saturation = p7 * strain / (1.0 + p8 * strain_abs**2)
    
    # Residual stress
    residual = p9
    
    # Combine all terms
    stress = (elastic + hardening + temp_effect + interaction 
              + cubic_correction + saturation + residual)
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
