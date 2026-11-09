"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a cubic polynomial in strain, temperature interactions, 
    and two non-linear saturation terms (tanh and exponential) sharing 
    a common rate parameter to capture elastic-plastic transition and 
    work hardening efficiently.
    
    params (10 constants):
      p0: Intercept
      p1: Linear strain coeff
      p2: Quadratic strain coeff
      p3: Cubic strain coeff
      p4: Temperature linear coeff
      p5: Temperature-strain interaction
      p6: Temperature squared coeff
      p7: Amplitude for tanh saturation
      p8: Common rate for saturation terms
      p9: Amplitude for exponential saturation
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Polynomial basis in strain (up to degree 3)
    eps2 = eps * eps
    eps3 = eps2 * eps
    
    # Temperature basis
    T2 = T * T
    
    # Non-linear terms
    # Ensure p8 is not zero to avoid division or flat functions if BFGS goes there,
    # though tanh(0)=0 and exp(0)=1 are well defined.
    rate = p8
    
    # Tanh saturation: smooth S-curve, good for initial yield
    sat_tanh = np.tanh(rate * eps)
    
    # Exponential saturation: captures hardening that plateaus
    # 1 - exp(-x) is ~x for small x, 1 for large x
    sat_exp = 1.0 - np.exp(-np.abs(rate) * eps)
    
    # Construct the full model
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps3
    sigma += p4 * T + p5 * T * eps + p6 * T2
    sigma += p7 * sat_tanh
    sigma += p9 * sat_exp
    
    return sigma
# EVOLVE-BLOCK-END
