"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Combines:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Linear response for population density (b)
    
    Uses raw inputs (no normalization) with parameters directly on the data scale.
    This allows the optimizer to find meaningful parameter values.
    
    Parameters (10 used):
    p0: Max growth rate scale (C)
    p1: Substrate half-saturation constant (K)
    p2: Optimal temperature (T_center)
    p3: Temperature response width (sigma_T)
    p4: Optimal pH (pH_center)
    p5: pH response width (sigma_pH)
    p6: Density coefficient (gamma)
    p7: Bias (B)
    p8: Additional substrate-temperature interaction
    p9: Additional substrate-pH interaction
    """
    # Unpack params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Monod substrate kinetics
    # Add small epsilon to prevent division by zero
    K = np.maximum(p1, 1e-6)
    monod = s / (K + s)

    # Temperature bell (Gaussian)
    # Use maximum to prevent division by zero and keep width positive
    sigma_T = np.maximum(np.abs(p3), 1.0)
    temp_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))

    # pH bell (Gaussian)
    sigma_pH = np.maximum(np.abs(p5), 1.0)
    ph_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))

    # Density term (linear inhibition)
    dens_term = 1.0 + p6 * b

    # Main multiplicative term
    main_term = p0 * monod * temp_term * ph_term * dens_term

    # Additive corrections
    correction = p7 + p8 * monod * (temp - p2) + p9 * monod * (pH - p4)

    result = main_term + correction

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
