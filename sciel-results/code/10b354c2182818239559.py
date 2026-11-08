"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod substrate limitation,
    Gaussian temperature and pH optima, and density inhibition.
    
    Parameters:
    b: population density, range [1, 4.998]
    s: substrate concentration, range [1, 4.999] (note: training range says up to 4.999, but inspiration says [1,40])
    temp: temperature, range [1.002, 39.98]
    pH: pH, range [2, 12]
    params: 10 parameters
        p[0]: mu_max (maximum growth rate)
        p[1]: K_s (Monod half-saturation constant)
        p[2]: T_opt (optimal temperature)
        p[3]: sigma_T (temperature width)
        p[4]: pH_opt (optimal pH)
        p[5]: sigma_pH (pH width)
        p[6]: K_b (density inhibition constant)
        p[7]: monod_exp (Monod exponent, for flexibility)
        p[8]: inh_exp (inhibition exponent)
        p[9]: offset
    """
    p = params
    
    # Extract parameters
    mu_max = p[0]
    K_s = p[1]
    T_opt = p[2]
    sigma_T = p[3]
    pH_opt = p[4]
    sigma_pH = p[5]
    K_b = p[6]
    monod_exp = p[7]
    inh_exp = p[8]
    offset = p[9]
    
    # Ensure positive values for denominators and variances to avoid NaNs
    eps = 1e-8
    
    # Use softplus-like or abs to ensure positivity where needed
    K_s_safe = np.log1p(np.exp(np.abs(K_s))) + eps
    sigma_T_safe = np.abs(sigma_T) + eps
    sigma_pH_safe = np.abs(sigma_pH) + eps
    K_b_safe = np.log1p(np.exp(np.abs(K_b))) + eps
    
    # Monod term for substrate
    # s is in [1, 4.999] in training (but might be wider in test)
    # Standard Monod: s / (Ks + s)
    # We raise to a power for flexibility
    monod_term = (s / (K_s_safe + s)) ** monod_exp
    
    # Temperature term (Gaussian)
    # temp in [1, 40], T_opt likely around 37
    temp_term = np.exp(-0.5 * ((temp - T_opt) / sigma_T_safe) ** 2)
    
    # pH term (Gaussian)
    # pH in [2, 12], pH_opt likely around 7
    ph_term = np.exp(-0.5 * ((pH - pH_opt) / sigma_pH_safe) ** 2)
    
    # Density inhibition term
    # b in [1, 5], inhibition should reduce growth at high density
    # Using 1 / (1 + (b / K_b)^inh_exp)
    # If K_b is small, inhibition is strong.
    b_term = (1.0 / (1.0 + (b / K_b_safe) ** inh_exp))
    
    # Combine all terms
    mu = mu_max * monod_term * temp_term * ph_term * b_term + offset
    
    return mu
# EVOLVE-BLOCK-END
