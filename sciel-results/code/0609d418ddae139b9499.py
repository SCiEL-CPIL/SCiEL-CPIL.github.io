"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses dual-tanh saturation (yield + hardening), linear elastic term,
    and polynomial temperature corrections with gated interaction.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Two tanh terms: one for yield plateau, one for hardening
    # p1 scales strain for the yield transition (sharp)
    # p2 scales strain for the hardening saturation (slower)
    s_yield = np.tanh(p1 * strain)
    s_hard = np.tanh(p2 * strain)
    
    # Linear strain for elastic region
    s1 = strain
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    
    # Gate: small when both tanh terms are saturated (deep plastic region)
    gate = (1.0 - np.abs(s_yield)) * (1.0 - np.abs(s_hard))
    
    # Build the model
    # p0: constant offset
    # p3 * s_yield: yield plateau contribution
    # p4 * s_hard: hardening saturation contribution
    # p5 * s1: linear elastic slope
    # p6 * t1: temperature linear
    # p7 * t2: temperature quadratic
    # p8 * s1 * t1: strain-temp interaction (gated)
    # p9 * s_yield * t1: yield-temperature interaction
    # p1 and p2 are scale factors for the tanh arguments (already used above)
    
    result = (p0
              + p3 * s_yield
              + p4 * s_hard
              + p5 * s1
              + p6 * t1
              + p7 * t2
              + p8 * s1 * t1 * gate
              + p9 * s_yield * t1)
    
    return result
# EVOLVE-BLOCK-END
