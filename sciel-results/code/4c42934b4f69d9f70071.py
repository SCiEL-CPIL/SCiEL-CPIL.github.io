import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Smooth Lorentzian-rational + polynomial form.
    The squared denominator guarantees C-infinity smoothness,
    providing a well-behaved loss landscape for BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Lorentzian-like saturation: E*eps / (1 + (a*eps)^2)
    # This smoothly transitions from linear elastic to saturated behavior
    a = p2 + p3 * temp
    denom = 1.0 + (a * strain) ** 2
    
    sigma_core = E * strain / denom
    
    # Polynomial correction terms with temperature modulation
    # These capture residual features the rational form misses
    poly2 = (p4 + p5 * temp) * strain ** 2
    poly3 = (p6 + p7 * temp) * strain ** 3
    poly4 = (p8 + p9 * temp) * strain ** 4
    
    return sigma_core + poly2 + poly3 + poly4
