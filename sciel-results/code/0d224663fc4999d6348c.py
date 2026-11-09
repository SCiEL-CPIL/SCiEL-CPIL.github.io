"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a flexible basis with multiple smooth nonlinear functions to capture
    elastic-plastic transition, work hardening, and temperature effects.
    
    params (10 constants):
      p0: Intercept
      p1: Linear strain coeff
      p2: Quadratic strain coeff
      p3: Cubic strain coeff
      p4: Quartic strain coeff
      p5: Temperature linear coeff
      p6: Temperature-strain interaction
      p7: Saturation amplitude
      p8: Saturation rate
      p9: Temperature-modulated saturation rate
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Polynomial basis in strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    
    # Temperature basis
    T2 = T * T
    
    # Smooth nonlinear saturation term: captures the elastic-to-plastic transition
    # Use tanh for smooth saturation
    sat_arg = (p8 + p9 * T) * eps
    sat = np.tanh(sat_arg)
    
    # Additional smooth term using arctan for a different saturation profile
    arctan_term = np.arctan((p8 + p9 * T) * eps)
    
    # Softplus-like term for work hardening
    # Use log(1 + exp(x)) which is smooth and captures gradual hardening
    x = (p8 + p9 * T) * eps
    softplus = np.log1p(np.exp(x))
    
    # Construct the full model with multiple nonlinear basis functions
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps3 + p4 * eps4
    sigma += p5 * T + p6 * T * eps
    sigma += p7 * sat
    sigma += 0.1 * p7 * arctan_term  # scaled to avoid dominance
    sigma += 0.05 * p7 * softplus     # scaled to avoid dominance
    
    return sigma
# EVOLVE-BLOCK-END
