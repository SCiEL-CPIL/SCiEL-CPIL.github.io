"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial basis in strain (up to 4th order) with temperature interactions.
    This allows the model to capture non-linear elastic/plastic behavior and
    temperature-dependent stiffness/hardening.
    """
    eps = strain
    T = temp
    
    # Ensure inputs are arrays
    eps = np.asarray(eps)
    T = np.asarray(T)
    
    # Basis functions
    e1 = eps
    e2 = eps * eps
    e3 = e2 * eps
    e4 = e2 * e2
    
    # Linear in params
    pred = (
        params[0] * np.ones_like(eps) +
        params[1] * e1 +
        params[2] * e2 +
        params[3] * e3 +
        params[4] * e4 +
        params[5] * T +
        params[6] * (e1 * T) +
        params[7] * (e2 * T) +
        params[8] * (e3 * T) +
        params[9] * (e4 * T)
    )
    
    return pred
# EVOLVE-BLOCK-END
