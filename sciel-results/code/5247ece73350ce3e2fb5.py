"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a dual-tanh model with fixed scales to capture yield and hardening,
    avoiding parameter correlation issues during BFGS optimization.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Pre-compute basis functions
    # Fixed scales for tanh to ensure distinct behaviors and stable gradients
    # Scale 10.0 captures the sharp yield transition (assuming yield is in the lower strain range)
    # Scale 0.5 captures the broader hardening plateau
    t_yield = np.tanh(10.0 * strain)
    t_hard = np.tanh(0.5 * strain)
    
    # Strain terms
    s1 = strain
    s2 = s1 * s1
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    
    # Build the model
    # p0: Constant offset
    # p1: Linear elastic slope
    # p2: Yield strength amplitude (sharp tanh)
    # p3: Hardening strength amplitude (broad tanh)
    # p4: Linear temperature shift
    # p5: Temperature-strain interaction (thermal expansion / modulus change)
    # p6: Temperature dependence of yield strength (interaction with sharp tanh)
    # p7: Quadratic temperature term
    # p8: Quadratic strain term (elastic curvature)
    # p9: High-order temperature-strain interaction
    
    result = (
        p0
        + p1 * s1
        + p2 * t_yield
        + p3 * t_hard
        + p4 * t1
        + p5 * t1 * s1
        + p6 * t1 * t_yield
        + p7 * t2
        + p8 * s2
        + p9 * t2 * s1
    )
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # Replace non-finite values with 0 to prevent fitter crash, 
        # though tanh and polynomials are generally stable
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
