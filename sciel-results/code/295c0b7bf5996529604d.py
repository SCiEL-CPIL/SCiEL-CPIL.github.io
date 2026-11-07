"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with:
    - Monod substrate saturation
    - Gaussian temperature optimum
    - Gaussian pH optimum
    - Exponential density inhibition
    - Linear correction terms for flexibility
    """
    p0 = params[0]  # max growth rate scale
    p1 = params[1]  # substrate half-saturation (Ks)
    p2 = params[2]  # optimal temperature
    p3 = params[3]  # temperature width
    p4 = params[4]  # optimal pH
    p5 = params[5]  # pH width
    p6 = params[6]  # density inhibition rate
    p7 = params[7]  # offset
    p8 = params[8]  # temp linear correction
    p9 = params[9]  # pH linear correction

    # Monod substrate saturation: s / (Ks + s)
    # Clamp Ks to avoid division by zero
    Ks = np.clip(p1, 1e-6, 1e6)
    substrate = s / (Ks + s)

    # Gaussian temperature factor
    # Clamp width to avoid division by zero
    width_t = np.clip(np.abs(p3), 1e-6, 1e6)
    temp_factor = np.exp(-((temp - p2) / width_t) ** 2)

    # Gaussian pH factor
    width_p = np.clip(np.abs(p5), 1e-6, 1e6)
    ph_factor = np.exp(-((pH - p4) / width_p) ** 2)

    # Exponential density inhibition
    # Clamp p6 to avoid extreme values
    p6_clamped = np.clip(p6, -10, 10)
    density_factor = np.exp(-p6_clamped * b)

    # Linear corrections for flexibility
    temp_corr = 1.0 + p8 * (temp - p2)
    ph_corr = 1.0 + p9 * (pH - p4)

    # Combine multiplicatively
    result = p0 * substrate * temp_factor * ph_factor * density_factor * temp_corr * ph_corr + p7

    return result
# EVOLVE-BLOCK-END
