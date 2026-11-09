"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack parameters
    p_max = params[0]
    K_s = params[1]
    T_opt_norm = params[2]
    sigma_T = params[3]
    pH_opt_norm = params[4]
    sigma_pH = params[5]
    K_b = params[6]
    
    # Normalize inputs to help optimization convergence
    # Temp range approx 1-40 C. Center at 20, scale by 10 -> range approx -1.9 to 2.0
    T_norm = (temp - 20.0) / 10.0
    # pH range approx 4-10. Center at 7, scale by 1 -> range approx -3 to 3
    pH_norm = pH - 7.0
    
    # Monod term for substrate
    # Avoid division by zero
    monod = s / (K_s + 1e-6 + s)
    
    # Temperature term (Gaussian)
    # Avoid division by zero in variance
    temp_term = np.exp(-0.5 * ((T_norm - T_opt_norm) ** 2) / (sigma_T ** 2 + 1e-6))
    
    # pH term (Gaussian)
    ph_term = np.exp(-0.5 * ((pH_norm - pH_opt_norm) ** 2) / (sigma_pH ** 2 + 1e-6))
    
    # Density inhibition term
    density_term = 1.0 / (1.0 + b / (K_b + 1e-6))
    
    # Final prediction
    y_pred = p_max * monod * temp_term * ph_term * density_term
    
    return y_pred
# EVOLVE-BLOCK-END
