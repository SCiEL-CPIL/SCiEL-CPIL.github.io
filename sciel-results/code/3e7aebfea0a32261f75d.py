"""Symbolic regression for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining:
    - Monod kinetics for substrate
    - Logistic/self-inhibition for density
    - Gaussian response for temperature
    - Quadratic response for pH
    
    params (10 constants):
      p0: max growth rate (mu_max)
      p1: Monod half-saturation (Ks)
      p2: density inhibition constant (Kb)
      p3: optimal temperature (T_opt)
      p4: temperature width (sigma_T)
      p5: optimal pH (pH_opt)
      p6: pH width (sigma_pH)
      p7: temperature amplitude (scale)
      p8: pH amplitude (scale)
      p9: baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive parameters where needed to avoid numerical issues
    Ks = np.abs(p1) + 1e-8
    Kb = np.abs(p2) + 1e-8
    sigma_T = np.abs(p4) + 1e-8
    sigma_pH = np.abs(p6) + 1e-8
    
    # Monod kinetics for substrate
    monod_s = s / (Ks + s)
    
    # Logistic self-inhibition for density
    logistic_b = b / (Kb + b)
    
    # Gaussian response for temperature
    gauss_T = np.exp(-0.5 * ((temp - p3) / sigma_T) ** 2)
    
    # Quadratic response for pH (bounded)
    quad_pH = 1.0 - ((pH - p5) / sigma_pH) ** 2
    # Clamp to avoid negative values
    quad_pH = np.maximum(quad_pH, 0.0)
    
    # Combine all factors
    growth = p0 * monod_s * logistic_b * (p7 * gauss_T) * (p8 * quad_pH) + p9
    
    return growth
# EVOLVE-BLOCK-END
