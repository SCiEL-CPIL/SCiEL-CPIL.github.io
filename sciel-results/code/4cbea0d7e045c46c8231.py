"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a fixed-basis additive model with polynomial and tanh terms.
    This creates a smooth, convex-like loss landscape for BFGS, allowing it to
    reliably find the global minimum and capture saturation behavior.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Compute basis functions
    # Polynomial terms in strain
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    
    # Tanh terms to capture saturation (plastic flow)
    # Scaling strain by 100 makes the transition sharper around 0
    t1 = np.tanh(100 * s1)
    t2 = np.tanh(100 * s1 - 1)  # Shifted tanh
    
    # Temperature terms
    t_temp1 = temp
    t_temp2 = temp * temp
    
    # Interaction terms
    st1 = s1 * t_temp1
    st2 = t1 * t_temp1
    
    # Build the model
    # We use all 10 parameters as coefficients for the basis functions
    # p0: intercept
    # p1: linear strain
    # p2: quadratic strain
    # p3: cubic strain
    # p4: tanh strain
    # p5: shifted tanh strain
    # p6: linear temp
    # p7: quadratic temp
    # p8: strain * temp
    # p9: tanh strain * temp
    
    prediction = (
        p0 +
        p1 * s1 +
        p2 * s2 +
        p3 * s3 +
        p4 * t1 +
        p5 * t2 +
        p6 * t_temp1 +
        p7 * t_temp2 +
        p8 * st1 +
        p9 * st2
    )
    
    return prediction
# EVOLVE-BLOCK-END
