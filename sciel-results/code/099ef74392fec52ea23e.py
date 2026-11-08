"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a physically-motivated model with polynomial terms in strain
    and temperature to capture non-linear stress-strain behavior.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Compute polynomial features
    # Strain terms
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    s4 = s2 * s2
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    
    # Interaction terms
    st1 = s1 * t1
    st2 = s2 * t1
    
    # Build the model as a weighted sum of features
    # Constant
    term_const = p0
    
    # Linear strain
    term_s1 = p1 * s1
    
    # Quadratic strain
    term_s2 = p2 * s2
    
    # Cubic strain
    term_s3 = p3 * s3
    
    # Quartic strain
    term_s4 = p4 * s4
    
    # Linear temperature
    term_t1 = p5 * t1
    
    # Quadratic temperature
    term_t2 = p6 * t2
    
    # Strain-temperature interaction (linear strain * linear temp)
    term_st1 = p7 * st1
    
    # Strain-temperature interaction (quadratic strain * linear temp)
    term_st2 = p8 * st2
    
    # Additional higher-order interaction
    term_s3t = p9 * (s3 * t1)
    
    # Sum all terms
    result = (term_const + term_s1 + term_s2 + term_s3 + term_s4 + 
              term_t1 + term_t2 + term_st1 + term_st2 + term_s3t)
    
    return result
# EVOLVE-BLOCK-END
