import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a generalized polynomial basis in strain and temperature.
    
    sigma = (p0 + p1*T)*u + (p2 + p3*T)*u^2 + (p4 + p5*T)*u^3 + (p6 + p7*T)*u^4 + (p8 + p9*T)
    
    This model is highly flexible for in-distribution data, allowing the optimizer
    to fit complex non-linear hardening curves and temperature interactions that
    may deviate from the standard Ramberg-Osgood power law.
    
    Parameters (10):
    p0, p1: Coefficients for linear strain term (Elastic + Linear Plastic)
    p2, p3: Coefficients for quadratic strain term
    p4, p5: Coefficients for cubic strain term
    p6, p7: Coefficients for quartic strain term
    p8, p9: Temperature-dependent offset (or constant term)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Precompute powers of strain
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    # Temperature-dependent coefficients
    c1 = p0 + p1 * T
    c2 = p2 + p3 * T
    c3 = p4 + p5 * T
    c4 = p6 + p7 * T
    c0 = p8 + p9 * T
    
    # Combine
    sigma = c1 * u + c2 * u2 + c3 * u3 + c4 * u4 + c0
    
    return sigma
