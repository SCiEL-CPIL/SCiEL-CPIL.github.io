import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a hyperbolic rational model with polynomial correction terms.
    The rational form provides a smooth elastic-to-plastic transition.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Hyperbolic term: captures yield point and initial hardening
    # Denominator: 1 + (p2^2 + 0.1) * strain * (1 + p3*temp)
    # Adding 0.1 to p2^2 ensures denominator is always > 1 for strain >= 0
    # This avoids singularities and keeps the function well-behaved
    denom = 1.0 + (p2 * p2 + 0.1) * strain * (1.0 + p3 * temp)
    
    hyper_term = (p0 * strain * (1.0 + p1 * temp)) / denom
    
    # Polynomial correction terms for hardening and temperature effects
    poly_term = (
        p4 * strain + 
        p5 * temp + 
        p6 * temp**2 + 
        p7 * strain * temp + 
        p8 + 
        p9 * strain**2
    )
    
    return hyper_term + poly_term
