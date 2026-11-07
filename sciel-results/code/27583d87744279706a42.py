import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Numerator: Temperature dependent linear and quadratic strain terms
    # (p0 + p1*T)*eps + (p2 + p3*T)*eps^2
    num = (p0 + p1 * T) * eps + (p2 + p3 * T) * (eps ** 2)
    
    # Denominator: 1 + (p4*eps^2)^2 + (p5*eps^4)^2
    # Ensures denominator >= 1, avoiding division by zero
    den = 1.0 + (p4 * (eps ** 2)) ** 2 + (p5 * (eps ** 4)) ** 2
    
    # Rational part
    rational = num / den
    
    # Linear/Polynomial part for temperature and residual effects
    # p6*eps + p7*T + p8*T*eps + p9*T^2
    poly = p6 * eps + p7 * T + p8 * T * eps + p9 * (T ** 2)
    
    return rational + poly
