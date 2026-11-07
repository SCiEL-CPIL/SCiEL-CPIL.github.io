"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    c0 = params[0]          # Max growth rate scale
    Ks = params[1]          # Monod half-saturation constant
    T_opt = params[2]       # Optimal temperature
    sigma_T = params[3]     # Temperature sensitivity width
    pH_opt = params[4]      # Optimal pH
    sigma_pH = params[5]    # pH sensitivity width
    alpha_b = params[6]     # Density-dependent inhibition strength
    beta_b = params[7]      # Density activation/saturation constant
    gamma = params[8]       # Cross-term coefficient (monod * temp)
    delta = params[9]       # Offset/baseline

    # Ensure positive denominators and widths for numerical stability
    Ks_safe = np.maximum(np.abs(Ks), 1e-4)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 0.5)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 0.5)
    alpha_b_safe = np.maximum(np.abs(alpha_b), 1e-4)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-4)

    # Monod substrate saturation: s / (Ks + s)
    monod = s / (Ks_safe + s)

    # Temperature Gaussian response: peaks at T_opt
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))

    # pH Gaussian response: peaks at pH_opt
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))

    # Density-dependent effects:
    # Inhibition term: 1 / (1 + alpha_b * b)  -> decreases with density
    density_inhibition = 1.0 / (1.0 + alpha_b_safe * b)
    # Activation/saturation term: b / (beta_b + b) -> increases with density, saturates
    density_activation = b / (beta_b_safe + b)

    # Combine: multiplicative core model
    growth = c0 * monod * temp_factor * ph_factor * density_inhibition

    # Add a secondary term that captures density activation interacting with substrate and temperature
    growth += gamma * monod * temp_factor * density_activation

    # Add a pH-modulated substrate term to capture pH-substrate interactions
    growth += delta * ph_factor * monod

    return growth
# EVOLVE-BLOCK-END
