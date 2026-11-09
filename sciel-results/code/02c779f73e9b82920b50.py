"""Rewritten program for E. coli growth rate symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a different structural family.
    
    Uses a piecewise-linear-in-log substrate term, asymmetric temperature response,
    and a rational density inhibition term.
    
    Model structure:
      mu = mu_max * (log1p(s) / (K_log + log1p(s))) * 
           exp(-((temp - T_opt) / (sigma_T + sigma_T_neg * (temp < T_opt)) / sigma_T)^2) *
           exp(-((pH - pH_opt) / sigma_pH)^2) *
           (K_b / (K_b + b)) + offset
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)

    mu_max = params[0]
    K_log = np.abs(params[1]) + 1e-8
    T_opt = params[2]
    sigma_T = np.abs(params[3]) + 1e-8
    sigma_T_neg = np.abs(params[4]) + 1e-8
    pH_opt = params[5]
    sigma_pH = np.abs(params[6]) + 1e-8
    K_b = np.abs(params[7]) + 1e-8
    alpha_b = np.abs(params[8]) + 0.1
    offset = params[9]

    # Log-saturation substrate term (different from Monod)
    log_s = np.log1p(np.clip(s, 0, None))
    substrate_factor = log_s / (K_log + log_s)

    # Asymmetric temperature response: wider on the low side if sigma_T_neg > sigma_T
    temp_diff = temp - T_opt
    # Effective sigma: use sigma_T on high side, sigma_T + sigma_T_neg on low side
    sigma_eff = np.where(temp_diff >= 0, sigma_T, sigma_T + sigma_T_neg)
    temp_factor = np.exp(-((temp_diff / sigma_eff) ** 2))

    # pH Gaussian
    ph_factor = np.exp(-((pH - pH_opt) / sigma_pH) ** 2)

    # Density inhibition with power law
    density_factor = (K_b / (K_b + b)) ** alpha_b

    # Combine
    mu = mu_max * substrate_factor * temp_factor * ph_factor * density_factor + offset

    return mu
# EVOLVE-BLOCK-END
