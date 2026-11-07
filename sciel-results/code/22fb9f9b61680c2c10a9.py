"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _softplus(x):
    """Smooth approximation of max(x, 0) using log(1+exp(x))."""
    # Numerically stable softplus
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _softabs(x):
    """Smooth approximation of |x| using sqrt(x^2 + eps)."""
    return np.sqrt(x * x + 1e-10)


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Biologically-motivated model for E. coli growth rate:
    - Monod kinetics for substrate saturation (with smooth Ks floor)
    - Gaussian responses for temperature and pH optima (with smooth sigma floors)
    - Linear term for biomass density effect
    All constraints handled with smooth surrogates to aid BFGS convergence.
    """
    # Unpack 10 parameters
    mu_max = params[0]      # Maximum growth rate
    Ks_raw = params[1]      # Monod half-saturation constant (raw)
    T_opt = params[2]       # Optimal temperature
    sigT_raw = params[3]    # Temperature width (raw)
    pH_opt = params[4]      # Optimal pH
    sigpH_raw = params[5]   # pH width (raw)
    c_b = params[6]         # Biomass density coefficient
    c_bs = params[7]        # Biomass-substrate interaction
    c_bt = params[8]        # Biomass-temperature interaction
    c_bp = params[9]        # Biomass-pH interaction
    
    # Normalize inputs to roughly [0, 1] range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Smooth positive floor for Ks: softplus(Ks_raw) ensures Ks > 0 smoothly
    Ks_safe = _softplus(Ks_raw) + 1e-8
    
    # Smooth positive floor for sigT: softabs(sigT_raw) ensures |sigT| > 0 smoothly
    sigT_safe = _softabs(sigT_raw) + 1e-8
    
    # Smooth positive floor for sigpH
    sigpH_safe = _softabs(sigpH_raw) + 1e-8
    
    # Monod kinetics for substrate: s / (Ks + s)
    # Use smooth denominator to avoid division issues
    monod_s = s_norm / (Ks_safe + s_norm + 1e-12)
    
    # Gaussian temperature response: exp(-((T - T_opt)^2) / (2 * sigT^2))
    temp_resp = np.exp(-((temp_norm - T_opt) ** 2) / (2.0 * sigT_safe ** 2))
    
    # Gaussian pH response: exp(-((pH - pH_opt)^2) / (2 * sigpH^2))
    ph_resp = np.exp(-((ph_norm - pH_opt) ** 2) / (2.0 * sigpH_safe ** 2))
    
    # Biomass density effect: linear with interactions
    b_effect = 1.0 + c_b * b_norm + c_bs * b_norm * s_norm + c_bt * b_norm * temp_norm + c_bp * b_norm * ph_norm
    
    # Combine all components
    mu = mu_max * monod_s * temp_resp * ph_resp * b_effect
    
    return mu
# EVOLVE-BLOCK-END
