"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a tanh-based model for the smooth elastic-plastic yield transition
    combined with a polynomial basis for strain hardening and temperature effects.
    
    Parameters:
    p0: Scale of the yield term
    p1: Sharpness of the yield transition
    p2: Temperature dependence of the yield transition
    p3: Linear strain correction
    p4: Linear temperature effect
    p5: Quadratic temperature effect
    p6: Strain-temperature interaction
    p7: Constant offset
    p8: Quadratic strain hardening
    p9: Cubic strain hardening
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are float64 arrays for numerical stability
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Smooth yield transition term
    # The factor (1.0 + p2 * T) allows the effective yield strain to shift with temperature
    # np.tanh provides a smooth, bounded transition from linear elastic to plastic plateau
    yield_term = p0 * np.tanh(p1 * eps * (1.0 + p2 * T))
    
    # Polynomial terms to capture hardening, thermal expansion/stress, and interactions
    poly_term = (
        p3 * eps + 
        p4 * T + 
        p5 * T**2 + 
        p6 * eps * T + 
        p7 + 
        p8 * eps**2 + 
        p9 * eps**3
    )
    
    return yield_term + poly_term
# EVOLVE-BLOCK-END
