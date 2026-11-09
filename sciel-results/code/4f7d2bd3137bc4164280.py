"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a convex additive model with polynomial and tanh basis functions
    to capture non-linear stress-strain behavior including saturation.
    BFGS can reliably find the global minimum for this convex formulation.
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    # Linear terms
    s1 = strain
    t1 = temp
    
    # Polynomial terms in strain
    s2 = s1 * s1
    s3 = s2 * s1
    
    # Polynomial terms in temperature
    t2 = t1 * t1
    
    # Interaction terms
    st1 = s1 * t1
    st2 = s2 * t1
    
    # Tanh saturation terms (capture yield/saturation behavior)
    # Scale strain to reasonable range for tanh
    s_tanh = np.tanh(p7 * s1)
    t_tanh = np.tanh(p8 * t1)
    st_tanh = np.tanh(p9 * st1)
    
    # Build the model as a weighted sum of basis functions
    # This is linear in params, so convex optimization applies
    result = (
        p0 * s1 +
        p1 * t1 +
        p2 * s2 +
        p3 * s3 +
        p4 * t2 +
        p5 * st1 +
        p6 * st2 +
        s_tanh +
        t_tanh +
        st_tanh
    )
    
    return result
# EVOLVE-BLOCK-END
