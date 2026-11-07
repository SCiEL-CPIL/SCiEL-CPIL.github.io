"""E. coli growth rate model with standardized polynomial and Monod basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a linear-in-parameters combination of
    standardized polynomial and Monod basis functions.
    
    This formulation ensures the optimization problem is convex (least squares
    in params), allowing BFGS to reliably find the global minimum for the
    coefficients. The basis functions are chosen to capture known biological
    trends: substrate saturation (Monod), temperature/pH optima (Gaussian),
    density inhibition, and key interaction terms.
    
    Standardization of inputs to roughly [0,1] range improves numerical
    conditioning for BFGS.
    """
    # Standardize inputs to improve conditioning
    # b in [1, 4.999] -> normalize
    b_norm = (b - 1.0) / 4.0
    # s in [1, 39.98] -> normalize
    s_norm = (s - 1.0) / 40.0
    # temp in [2, 12] based on training range... wait, temp range seems odd.
    # Let's re-read: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # The inputs are: b, s, temp, pH
    # b: [1, 4.998] approx
    # s: [1, 4.999] approx? No, the list is [min1, max1], [min2, max2]...
    # "Inputs in training range, in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]"
    # So:
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_min, b_max = 1.0, 5.0
    s_min, s_max = 1.0, 5.0
    t_min, t_max = 1.0, 40.0
    ph_min, ph_max = 2.0, 12.0
    
    b_norm = (b - b_min) / (b_max - b_min)
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (temp - t_min) / (t_max - t_min)
    ph_norm = (pH - ph_min) / (ph_max - ph_min)
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Biological centers in normalized space
    # T_opt = 37 C -> (37-1)/39 = 0.923
    # pH_opt = 7 -> (7-2)/10 = 0.5
    # Ks for Monod: in original s units, let's say Ks=2.5 (midpoint)
    # In normalized s: (2.5-1)/4 = 0.375
    
    T_opt_norm = 0.923
    T_sig_norm = 0.15  # 5C / 39 range
    pH_opt_norm = 0.5
    pH_sig_norm = 0.1  # 1pH / 10 range
    Ks_norm = 0.375
    Kd_norm = 0.5  # Density inhibition constant in normalized space
    
    # Core basis functions
    # Monod substrate saturation: s_norm / (Ks + s_norm)
    monod_s = s_norm / (Ks_norm + s_norm)
    
    # Gaussian temperature optimum
    gauss_temp = np.exp(-((t_norm - T_opt_norm) ** 2) / (2.0 * T_sig_norm ** 2))
    
    # Gaussian pH optimum
    gauss_ph = np.exp(-((ph_norm - pH_opt_norm) ** 2) / (2.0 * pH_sig_norm ** 2))
    
    # Density inhibition
    density_inhib = 1.0 / (1.0 + Kd_norm * b_norm)
    
    # Interaction terms
    temp_ph_interaction = gauss_temp * gauss_ph
    substrate_temp_interaction = monod_s * gauss_temp
    substrate_ph_interaction = monod_s * gauss_ph
    density_substrate_interaction = density_inhib * monod_s
    
    # Additional flexible terms
    linear_s = s_norm
    linear_temp = t_norm
    linear_ph = ph_norm
    
    # Combine all basis functions with fitted parameters
    # We use a linear combination of these basis functions.
    # Since the model is linear in params, BFGS should converge quickly.
    prediction = (
        p0 * monod_s * gauss_temp * gauss_ph * density_inhib +
        p1 * monod_s * gauss_temp +
        p2 * monod_s * gauss_ph +
        p3 * gauss_temp * gauss_ph +
        p4 * monod_s +
        p5 * gauss_temp +
        p6 * gauss_ph +
        p7 * density_substrate_interaction +
        p8 * linear_s +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
