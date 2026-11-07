import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a full cubic bivariate polynomial expansion (Taylor series up to degree 3).
    Terms: 1, s, t, s^2, st, t^2, s^3, s^2t, st^2, t^3
    """
    s = strain
    t = temp
    
    # Precompute powers
    s2 = s * s
    s3 = s2 * s
    t2 = t * t
    t3 = t2 * t
    
    # Construct the polynomial
    # p0: constant
    # p1: linear strain
    # p2: linear temp
    # p3: quadratic strain
    # p4: interaction
    # p5: quadratic temp
    # p6: cubic strain
    # p7: cubic interaction (s^2 t)
    # p8: cubic interaction (s t^2)
    # p9: cubic temp
    
    return (params[0] + 
            params[1] * s + 
            params[2] * t + 
            params[3] * s2 + 
            params[4] * s * t + 
            params[5] * t2 + 
            params[6] * s3 + 
            params[7] * s2 * t + 
            params[8] * s * t2 + 
            params[9] * t3)
