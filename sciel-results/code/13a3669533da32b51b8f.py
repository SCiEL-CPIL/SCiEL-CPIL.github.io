"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a flexible additive basis expansion (polynomial + interactions) to capture
    complex non-linear relationships between strain, temperature, and stress.
    
    10 parameters corresponding to basis functions:
    p0: constant
    p1: strain (linear elastic)
    p2: strain^2
    p3: strain^3
    p4: temp (linear temp effect)
    p5: temp^2
    p6: strain * temp (interaction)
    p7: strain^2 * temp (interaction)
    p8: temp^3
    p9: strain * temp^2 (interaction)
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    s2 = strain * strain
    s3 = s2 * strain
    t2 = temp * temp
    t3 = t2 * temp
    
    # Basis expansion
    # Note: Coefficients are free to be negative/positive, allowing for complex curves
    # This structure allows for yield-like behavior if p2/p3 are negative and p1 is positive,
    # or other shapes depending on the fit.
    
    prediction = (
        p0 + 
        p1 * strain + 
        p2 * s2 + 
        p3 * s3 + 
        p4 * temp + 
        p5 * t2 + 
        p6 * strain * temp + 
        p7 * s2 * temp + 
        p8 * t3 + 
        p9 * strain * t2
    )
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
