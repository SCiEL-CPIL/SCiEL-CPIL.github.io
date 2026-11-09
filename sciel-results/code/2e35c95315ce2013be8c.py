"""Improved symbolic regression model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative temperature-modulated elastic-plastic model with sinh hardening.
    
    Model structure:
      E(T) = p0 * (1 + p1*T + p2*T^2)     # temperature-dependent elastic modulus
      sigma_y(T) = p3 * (1 + p4*T)         # temperature-dependent yield stress
      hardening(eps, T) = sigma_y(T) * sinh(p5 * eps) / (1 + p6 * sinh(p5 * eps)^2)
      stress = E(T) * eps * (1 + p7*eps + p8*eps^2) + hardening(eps, T) * (1 + p9*eps)
    
    The multiplicative coupling between E(T) and strain terms allows the
    temperature to directly modulate both elastic response and hardening rate.
    All 10 params are fitted by BFGS.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus (low-order polynomial)
    E_T = p0 * (1.0 + p1 * temp + p2 * temp * temp)
    
    # Temperature-dependent yield stress
    sigma_y_T = p3 * (1.0 + p4 * temp)
    
    # Compute sinh argument, clip to avoid overflow
    sinh_arg = p5 * strain
    sinh_arg = np.clip(sinh_arg, -500, 500)
    
    # sinh-based hardening: grows then saturates
    sinh_val = np.sinh(sinh_arg)
    # Normalize to prevent overflow: sinh(x)/(1+a*sinh^2(x)) behaves like x for small x,
    # and saturates for large |x|
    denom = 1.0 + p6 * sinh_val * sinh_val
    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10 + 1e-10, denom)
    hardening = sigma_y_T * sinh_val / denom
    
    # Elastic component with strain-dependent correction (multiplicative)
    eps2 = strain * strain
    elastic_correction = 1.0 + p7 * strain + p8 * eps2
    elastic = E_T * strain * elastic_correction
    
    # Hardening component with additional strain modulation
    hardening_modulated = hardening * (1.0 + p9 * strain)
    
    prediction = elastic + hardening_modulated
    
    return prediction
# EVOLVE-BLOCK-END
