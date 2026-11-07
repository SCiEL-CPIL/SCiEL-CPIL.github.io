"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with Monod kinetics, temperature/pH optima,
    and density-dependent inhibition.
    
    Model: mu = p0 * (s / (p1 + s)) * exp(-((temp - p2)^2) / (2*p3^2))
                        * exp(-((pH - p4)^2) / (2*p5^2))
                        * (1 + p6*b)^(-p7)
                        + p8 * (temp - p2) * (pH - p4) + p9
    
    Parameters (10 total):
        p0: Maximum specific growth rate
        p1: Monod half-saturation constant (Ks)
        p2: Optimal temperature
        p3: Temperature width (sigma_T)
        p4: Optimal pH
        p5: pH width (sigma_pH)
        p6: Density inhibition constant
        p7: Density inhibition exponent
        p8: Interaction coefficient (temp-pH coupling)
        p9: Baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod substrate term: s / (Ks + s), Ks = exp(p1) to keep positive
    Ks = np.exp(p1)
    monod = s / (Ks + s)
    
    # Temperature response: Gaussian around optimal temp
    # Use exp to ensure positive width
    sigma_T = np.exp(p3)
    temp_factor = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH response: Gaussian around optimal pH
    sigma_pH = np.exp(p5)
    ph_factor = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density-dependent inhibition: (1 + Kb * b)^(-n)
    # Kb = exp(p6) to keep positive
    Kb = np.exp(p6)
    density_factor = (1.0 + Kb * b) ** (-p7)
    
    # Main multiplicative term
    mu = p0 * monod * temp_factor * ph_factor * density_factor
    
    # Add interaction term for temp-pH coupling
    interaction = p8 * (temp - p2) * (pH - p4)
    
    # Final prediction
    return mu + interaction + p9
# EVOLVE-BLOCK-END
