import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack 10 parameters
    c0 = params[0]          # Max growth rate scale
    Ks = params[1]          # Monod half-saturation constant
    T_opt = params[2]       # Optimal temperature
    sigma_T = params[3]     # Temperature sensitivity width
    pH_opt = params[4]      # Optimal pH
    sigma_pH = params[5]    # pH sensitivity width
    alpha_b = params[6]     # Density-dependent inhibition strength
    beta_b = params[7]      # Density activation/saturation constant
    gamma = params[8]       # Cross-term coefficient
    delta = params[9]       # Offset/baseline

    # Ensure positive denominators and widths for numerical stability
    Ks_safe = np.maximum(np.abs(Ks), 1e-4)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 0.5)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 0.5)
    alpha_b_safe = np.maximum(np.abs(alpha_b), 1e-6)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-6)

    # Monod substrate saturation: s / (Ks + s)
    monod = s / (Ks_safe + s)

    # Temperature Gaussian response: peaks at T_opt
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))

    # pH Gaussian response: peaks at pH_opt
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))

    # Density-dependent effects:
    # Inhibition: 1 / (1 + alpha_b * b) -> decreases with density
    density_inhibition = 1.0 / (1.0 + alpha_b_safe * b)
    # Activation/saturation: b / (beta_b + b) -> increases with density, saturates
    density_activation = b / (beta_b_safe + b)

    # Combine: multiplicative core model with both density effects
    # Core: c0 * monod * temp * ph * inhibition
    # Secondary: gamma * monod * temp * activation (captures density-activation interaction)
    growth = c0 * monod * temp_factor * ph_factor * density_inhibition
    growth += gamma * monod * temp_factor * density_activation

    # Baseline offset
    growth += delta

    return growth
