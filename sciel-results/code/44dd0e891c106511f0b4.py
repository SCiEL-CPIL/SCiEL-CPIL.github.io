"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses temperature-modulated yield saturation and power-law strain hardening.
    
    params (10 constants):
      p0: Intercept
      p1: Linear strain coeff (elastic slope)
      p2: Quadratic strain coeff
      p3: Cubic strain coeff
      p4: Temperature linear coeff (yield drop)
      p5: Temperature-strain interaction
      p6: Saturation amplitude
      p7: Saturation rate (yield strain scale)
      p8: Strain hardening exponent
      p9: Hardening amplitude
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Polynomial basis in strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    
    # Temperature effect on yield: yield strength decreases with T
    # Modulate the saturation term with temperature
    T_factor = 1.0 + p4 * T + p5 * T * eps
    
    # Saturation term: smooth transition from elastic to plastic
    # Use tanh for smooth S-curve, rate modulated by temperature
    rate = np.abs(p7) + 1e-8  # avoid zero rate
    sat = np.tanh(rate * eps)
    
    # Power-law strain hardening: eps^alpha captures decaying hardening
    # Clip eps to avoid issues with negative values raised to non-integer powers
    eps_clipped = np.maximum(eps, 0.0)
    hardening = eps_clipped ** p8
    
    # Construct the full model
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps3
    sigma += p6 * T_factor * sat
    sigma += p9 * hardening
    
    return sigma
# EVOLVE-BLOCK-END
