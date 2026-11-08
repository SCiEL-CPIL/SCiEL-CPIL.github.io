"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a tanh-saturation model for strain hardening with temperature corrections.
    """
    e = strain
    t = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core elastic-plastic response with saturation
    # p0: initial yield offset
    # p1: elastic modulus scale
    # p2: saturation strain scale (inverse)
    # p3: saturation stress scale
    elastic = p1 * e
    hardening = p3 * np.tanh(p2 * np.maximum(e, 0.0))
    
    # Temperature corrections
    # p4: thermal stress coefficient
    # p5: temp-modulated yield shift
    # p6: temp-modulated hardening rate
    temp_effect = p4 * (t - 0.5)
    temp_yield = p5 * (t - 0.5) * e
    temp_hard = p6 * (t - 0.5) * np.tanh(p2 * np.maximum(e, 0.0))
    
    # Higher-order corrections for nonlinearity
    quad = p7 * e**2 * (1.0 - np.tanh(p2 * np.maximum(e, 0.0)))
    cubic = p8 * e**3 * (1.0 - np.tanh(p2 * np.maximum(e, 0.0)))
    cross = p9 * t * e**2
    
    return (p0 + elastic + hardening + temp_effect + temp_yield + 
            temp_hard + quad + cubic + cross)
# EVOLVE-BLOCK-END
