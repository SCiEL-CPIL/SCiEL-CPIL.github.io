"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    c0, c1, Ks, muT, sigT, mupH, sigpH, alpha_b, beta_b, h = params
    
    # Safe denominators
    Ks_safe = np.maximum(np.abs(Ks), 1e-6)
    sigT_safe = np.maximum(np.abs(sigT), 1e-6)
    sigpH_safe = np.maximum(np.abs(sigpH), 1e-6)
    h_safe = np.maximum(np.abs(h), 1e-6)
    
    # Monod-like substrate term
    s_term = s / (Ks_safe + s)
    
    # Temperature Gaussian
    T_term = np.exp(-((temp - muT) ** 2) / (2.0 * sigT_safe ** 2))
    
    # pH Gaussian
    pH_term = np.exp(-((pH - mupH) ** 2) / (2.0 * sigpH_safe ** 2))
    
    # Bacterial density term (inhibition or activation)
    b_term = b / (alpha_b + b)
    
    # Combined multiplicative model
    return c0 * s_term * T_term * pH_term * b_term + c1 * b + h_safe
# EVOLVE-BLOCK-END
