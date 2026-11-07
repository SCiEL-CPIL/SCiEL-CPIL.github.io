"""E. coli growth rate model with standardized polynomial and Monod basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a linear combination of standardized
    biological basis functions. The basis is fixed (non-parametric in structure)
    and the 10 parameters are linear coefficients, making the optimization
    a convex least-squares problem that BFGS can solve reliably.
    
    Basis functions:
      - Monod substrate saturation
      - Gaussian temperature response
      - Gaussian pH response
      - Density inhibition
      - Key interaction terms
      - Constant
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological centers (based on E. coli physiology)
    Ks = 5.0        # Substrate half-saturation
    T_opt = 37.0    # Optimal temperature
    T_sig = 5.0     # Temperature sensitivity
    pH_opt = 7.0    # Optimal pH
    pH_sig = 1.0    # pH sensitivity
    Kd = 1.0        # Density inhibition constant
    
    # Core basis functions
    monod_s = s / (Ks + s)
    gauss_temp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_sig ** 2))
    gauss_ph = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_sig ** 2))
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Normalized inputs for polynomial terms (keep in stable range)
    b_norm = np.clip((b - 1.0) / (5.0 - 1.0), -1.0, 1.0)
    s_norm = np.clip((s - 1.0) / (40.0 - 1.0), -1.0, 1.0)
    temp_norm = np.clip((temp - 37.0) / (5.0), -1.0, 1.0)
    ph_norm = np.clip((pH - 7.0) / (1.0), -1.0, 1.0)
    
    # Polynomial interaction terms for flexibility
    temp_quad = temp_norm ** 2
    ph_quad = ph_norm ** 2
    temp_ph_cross = temp_norm * ph_norm
    s_temp_cross = s_norm * temp_norm
    s_ph_cross = s_norm * ph_norm
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * monod_s * gauss_temp * gauss_ph * density_inhib +
        p1 * monod_s * gauss_temp +
        p2 * monod_s * gauss_ph +
        p3 * gauss_temp * gauss_ph +
        p4 * monod_s +
        p5 * gauss_temp +
        p6 * gauss_ph +
        p7 * temp_quad +
        p8 * ph_quad +
        p9 * temp_ph_cross
    )
    
    return prediction
# EVOLVE-BLOCK-END
