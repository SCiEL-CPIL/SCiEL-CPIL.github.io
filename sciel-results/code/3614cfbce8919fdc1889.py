import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a hyperbolic model for the elastic-plastic transition
    combined with polynomial terms for hardening and temperature effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays for broadcasting
    strain = np.asarray(strain)
    temp = np.asarray(temp)
    
    # Hyperbolic term: captures the yield point and initial hardening
    # Denominator uses p2^2 to ensure it contributes positively to 1, avoiding singularities
    # from the wrong side, although BFGS should avoid 0 anyway.
    denom = 1.0 + (p2 * p2) * strain * (1.0 + p3 * temp)
    # To be absolutely safe against division by zero if params go weird,
    # we can add a tiny epsilon, but this might bias the fit slightly.
    # Given the 1e30 penalty, standard division is usually fine if the loss
    # function guides away from 0.
    
    hyper_term = (p0 * strain * (1.0 + p1 * temp)) / denom
    
    # Polynomial terms: capture linear elasticity, plastic hardening, 
    # temperature dependence, and interactions.
    poly_term = (
        p4 * strain + 
        p5 * temp + 
        p6 * temp**2 + 
        p7 * strain * temp + 
        p8 + 
        p9 * strain**3
    )
    
    return hyper_term + poly_term
