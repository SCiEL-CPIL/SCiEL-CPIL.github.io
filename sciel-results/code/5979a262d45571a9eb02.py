"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a physically-motivated model with tanh saturation for strain hardening,
    polynomial temperature corrections, and gated interaction terms.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Scale strain to a reasonable range for tanh saturation
    # Typical strain range is [-0.002, ~1], so scale by ~2 to get tanh working well
    s_scaled = p1 * strain
    
    # Tanh saturation for elastic-plastic transition and hardening
    # This captures the initial linear elastic region and the nonlinear hardening
    hardening = np.tanh(s_scaled)
    
    # Linear strain term (elastic region)
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    t3 = t2 * t1
    
    # Gating: (1 - tanh) is small where saturation has occurred
    # This reduces correlation and keeps correction terms bounded
    gate = 1.0 - np.abs(np.tanh(s_scaled))
    
    # Build the model
    # Constant offset
    term_const = p0
    
    # Primary hardening term (tanh captures the S-shape of stress-strain)
    term_hardening = p2 * hardening
    
    # Linear strain correction (for the elastic slope)
    term_s1 = p3 * s1
    
    # Quadratic strain (curvature in elastic region)
    term_s2 = p4 * s2
    
    # Cubic strain (additional nonlinearity)
    term_s3 = p5 * s3
    
    # Temperature linear
    term_t1 = p6 * t1
    
    # Temperature quadratic
    term_t2 = p7 * t2
    
    # Temperature cubic
    term_t3 = p8 * t3
    
    # Gated strain-temperature interaction
    # Gate keeps this term small in the saturated (plastic) region
    term_st = p9 * s1 * t1 * gate
    
    # Sum all terms
    result = (term_const + term_hardening + term_s1 + term_s2 + term_s3 + 
              term_t1 + term_t2 + term_t3 + term_st)
    
    return result
# EVOLVE-BLOCK-END
