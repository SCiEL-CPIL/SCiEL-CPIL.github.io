"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a model combining linear, quadratic, and saturating exponential terms
    with temperature-dependent coefficients. This approach captures elastic,
    hardening, and saturation behaviors distinctly from rational functions.
    """
    s = strain
    t = temp
    
    # Extract parameters
    # p0, p1: Linear term coefficients (p0 + p1*T)
    # p2, p3: Quadratic term coefficients (p2 + p3*T)
    # p4, p5: Exponential amplitude (p4 + p5*T)
    # p6, p7: Exponential rate (p6 + p7*T)
    # p8, p9: Offset (p8 + p9*T)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute temperature-dependent coefficients
    c_lin = p0 + p1 * t
    c_quad = p2 + p3 * t
    c_exp_amp = p4 + p5 * t
    c_exp_rate = p6 + p7 * t
    c_offset = p8 + p9 * t
    
    # Compute terms
    term_linear = c_lin * s
    term_quadratic = c_quad * s * s
    term_exponential = c_exp_amp * (1.0 - np.exp(-c_exp_rate * s))
    
    # Sum terms
    stress = c_offset + term_linear + term_quadratic + term_exponential
    
    return stress
# EVOLVE-BLOCK-END
