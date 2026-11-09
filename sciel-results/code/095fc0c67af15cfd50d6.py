"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a biologically-inspired multiplicative model with Monod kinetics,
    Gaussian temperature response, and pH response, plus interaction terms.
    """
    # Unpack parameters
    p_max = params[0]      # Maximum growth rate
    K_s = params[1]        # Half-saturation constant for substrate
    T_opt = params[2]      # Optimal temperature
    sigma_T = params[3]    # Temperature tolerance
    pH_opt = params[4]     # Optimal pH
    sigma_pH = params[5]   # pH tolerance
    K_b = params[6]        # Half-saturation for population density (inhibition)
    alpha_s = params[7]    # Substrate-temperature interaction
    alpha_pH = params[8]   # pH-temperature interaction
    bias = params[9]       # Bias term
    
    # Safe parameters to avoid numerical issues
    K_s_safe = np.maximum(K_s, 1e-8)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 1e-8)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 1e-8)
    K_b_safe = np.maximum(np.abs(K_b), 1e-8)
    
    # Monod term for substrate: s / (K_s + s)
    monod = s / (K_s_safe + s)
    
    # Temperature response: Gaussian
    T_diff = temp - T_opt
    temp_response = np.exp(-0.5 * (T_diff / sigma_T_safe) ** 2)
    
    # pH response: Gaussian
    pH_diff = pH - pH_opt
    pH_response = np.exp(-0.5 * (pH_diff / sigma_pH_safe) ** 2)
    
    # Population density effect: logistic-type, allowing for both promotion and inhibition
    # b / (K_b + b) saturates at 1, or we can use 1 - b/(K_b+b) for inhibition
    # Let's use a flexible form: (1 + alpha_b * b) / (1 + b/K_b)
    # But simpler: use b/(K_b+b) which goes from 0 to 1
    b_term = b / (K_b_safe + b)
    
    # Interaction terms
    # These allow the temperature optimum to shift with substrate or pH
    T_shift_s = alpha_s * monod
    T_shift_pH = alpha_pH * pH_response
    
    # Adjusted temperature difference
    T_diff_adj = temp - (T_opt + T_shift_s + T_shift_pH)
    temp_response_adj = np.exp(-0.5 * (T_diff_adj / sigma_T_safe) ** 2)
    
    # Combine all factors multiplicatively
    # Growth rate = p_max * monod * temp_response * pH_response * b_term + bias
    growth = p_max * monod * temp_response_adj * pH_response * b_term + bias
    
    # Ensure finite output
    return np.where(np.isfinite(growth), growth, 0.0)
# EVOLVE-BLOCK-END
