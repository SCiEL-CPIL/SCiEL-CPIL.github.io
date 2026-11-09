"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack parameters
    p_max_raw = params[0]
    K_s_raw = params[1]
    T_opt_norm = params[2]
    sigma_T_raw = params[3]
    pH_opt_norm = params[4]
    sigma_pH_raw = params[5]
    K_b_raw = params[6]
    base_rate = params[7]
    T_pH_interact = params[8]
    exponent_T = params[9]
    
    # Ensure positive parameters via softplus-like transform
    p_max = np.exp(p_max_raw)
    K_s = np.exp(K_s_raw)
    sigma_T = np.exp(sigma_T_raw)
    sigma_pH = np.exp(sigma_pH_raw)
    K_b = np.exp(K_b_raw)
    
    # Normalize inputs to help optimization convergence
    # Temp range approx 1-40 C. Center at 20, scale by 10 -> range approx -1.9 to 2.0
    T_norm = (temp - 20.0) / 10.0
    # pH range approx 4-10. Center at 7, scale by 1 -> range approx -3 to 3
    pH_norm = pH - 7.0
    
    # Monod term for substrate
    monod = s / (K_s + s)
    
    # Temperature term (Gaussian) with adjustable width
    temp_term = np.exp(-0.5 * ((T_norm - T_opt_norm) ** 2) / (sigma_T ** 2))
    
    # pH term (Gaussian)
    ph_term = np.exp(-0.5 * ((pH_norm - pH_opt_norm) ** 2) / (sigma_pH ** 2))
    
    # Temperature-pH interaction term
    T_pH_term = np.exp(T_pH_interact * T_norm * pH_norm)
    
    # Density inhibition term
    density_term = 1.0 / (1.0 + b / K_b)
    
    # Final prediction with baseline
    y_pred = base_rate + p_max * monod * temp_term * ph_term * T_pH_term * density_term
    
    return y_pred
# EVOLVE-BLOCK-END
