"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Biologically-motivated model for E. coli growth rate:
    - Monod kinetics for substrate saturation
    - Gaussian responses for temperature and pH optima
    - Linear term for biomass density effect
    """
    # Unpack 10 parameters
    mu_max = params[0]      # Maximum growth rate
    Ks = params[1]          # Monod half-saturation constant
    T_opt = params[2]       # Optimal temperature
    sigT = params[3]        # Temperature width
    pH_opt = params[4]      # Optimal pH
    sigpH = params[5]       # pH width
    c_b = params[6]         # Biomass density coefficient
    c_bs = params[7]        # Biomass-substrate interaction
    c_bt = params[8]        # Biomass-temperature interaction
    c_bp = params[9]        # Biomass-pH interaction
    
    # Normalize inputs to roughly [0, 1] range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Monod kinetics for substrate: s / (Ks + s)
    # Ensure Ks > 0 for numerical stability
    Ks_safe = np.maximum(Ks, 1e-6)
    monod_s = s_norm / (Ks_safe + s_norm)
    
    # Gaussian temperature response: exp(-((T - T_opt)^2) / (2 * sigT^2))
    sigT_safe = np.maximum(np.abs(sigT), 1e-6)
    temp_resp = np.exp(-((temp_norm - T_opt) ** 2) / (2.0 * sigT_safe ** 2))
    
    # Gaussian pH response: exp(-((pH - pH_opt)^2) / (2 * sigpH^2))
    sigpH_safe = np.maximum(np.abs(sigpH), 1e-6)
    ph_resp = np.exp(-((ph_norm - pH_opt) ** 2) / (2.0 * sigpH_safe ** 2))
    
    # Biomass density effect: linear with interactions
    b_effect = 1.0 + c_b * b_norm + c_bs * b_norm * s_norm + c_bt * b_norm * temp_norm + c_bp * b_norm * ph_norm
    
    # Combine all components
    mu = mu_max * monod_s * temp_resp * ph_resp * b_effect
    
    return mu
# EVOLVE-BLOCK-END
