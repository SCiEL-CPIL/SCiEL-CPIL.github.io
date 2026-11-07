"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with biologically motivated terms:
    - Monod saturation for substrate
    - Gaussian temperature optimum
    - Gaussian pH optimum
    - Density-dependent inhibition (quadratic)
    - Interaction between substrate and density
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Monod substrate term: s / (Ks + s), Ks = p1
    # Ensure Ks > 0 for numerical stability
    Ks = p1
    monod = s / (Ks + s)

    # Temperature Gaussian: exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    T_opt = p2
    sigma_T = p3
    # Avoid division by zero
    sigma_T_safe = sigma_T * sigma_T + 1e-6
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe))

    # pH Gaussian: exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    pH_opt = p4
    sigma_pH = p5
    sigma_pH_safe = sigma_pH * sigma_pH + 1e-6
    pH_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe))

    # Density inhibition: 1 / (1 + (b / K_b)^2) where K_b = p7
    # This gives 1 at b=0 and decreases as b increases
    K_b = p7
    density_inhibit = 1.0 / (1.0 + (b / K_b) ** 2)

    # Combine all factors
    # Base growth: amplitude * monod * temp_factor * pH_factor * density_inhibit
    base = p0 * monod * temp_factor * pH_factor * density_inhibit

    # Add linear correction terms for flexibility
    # These allow the model to deviate from the pure multiplicative form
    correction = p6 * b + p8 * s + p9 * (temp - T_opt) * (pH - pH_opt)

    return base + correction
# EVOLVE-BLOCK-END
