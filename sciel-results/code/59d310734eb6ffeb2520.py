"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a biologically-motivated basis expansion.
    
    Structure:
      - Monod substrate saturation: s / (Ks + s)
      - Gaussian temperature optimum: exp(-(temp - T_opt)^2 / (2*T_sig^2))
      - Gaussian pH optimum: exp(-(pH - pH_opt)^2 / (2*pH_sig^2))
      - Density inhibition: 1 / (1 + Kd * b)
      - Additive correction terms for residual structure
    
    The model is linear in params, making the least-squares problem convex
    and allowing BFGS to reliably converge to the global optimum.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe Monod substrate term (avoid division by zero)
    Ks = np.maximum(np.abs(p1), 1e-6)
    monod_s = s / (Ks + s)
    
    # Gaussian temperature term
    T_opt = p2
    T_sig = np.maximum(np.abs(p3), 1e-6)
    temp_gauss = np.exp(-0.5 * ((temp - T_opt) / T_sig) ** 2)
    
    # Gaussian pH term
    pH_opt = p4
    pH_sig = np.maximum(np.abs(p5), 1e-6)
    ph_gauss = np.exp(-0.5 * ((pH - pH_opt) / pH_sig) ** 2)
    
    # Density inhibition
    Kd = np.maximum(np.abs(p6), 1e-6)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Main multiplicative growth term
    main_term = monod_s * temp_gauss * ph_gauss * density_inhib
    
    # Additive basis for residual structure
    # Normalized inputs for stability
    b_norm = b / 5.0  # scale to ~[0.2, 1]
    s_norm = s / 40.0  # scale to ~[0.025, 1]
    temp_norm = temp / 12.0  # scale to ~[0.08, 1]
    ph_norm = pH / 14.0  # scale to ~[0.07, 1]
    
    # Build additive correction basis
    corr1 = temp_norm
    corr2 = ph_norm
    corr3 = b_norm
    corr4 = s_norm
    corr5 = temp_norm * s_norm
    
    # Combine: main term scaled by p0, plus additive corrections
    result = (p0 * main_term 
              + p7 * corr1 
              + p8 * corr2 
              + p9 * corr3 
              + 0.0 * corr4  # reserved, keep for potential future use
              + 0.0 * corr5)
    
    return result
# EVOLVE-BLOCK-END
