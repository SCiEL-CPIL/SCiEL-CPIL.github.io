import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Linear Elastic + Rational Hardening
    sigma = E(T)*eps + (H(T)*eps^2 + C(T)*eps^3) / (1 + K1(T)*eps + K2(T)*eps^2)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Elastic Modulus E(T) = p0 + p1*T
    E = p0 + p1 * temp
    
    # Hardening Numerator Coeffs
    H = p2 + p3 * temp
    C = p8 + p9 * temp  # Cubic term coeff
    
    # Hardening Denominator Coeffs
    K1 = p4 + p5 * temp
    K2 = p6 + p7 * temp
    
    # Calculate terms
    eps = strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    
    numerator = H * eps2 + C * eps3
    denominator = 1.0 + K1 * eps + K2 * eps2
    
    # Safety guard against division by zero or near-zero
    # If denominator is close to 0, the model is unstable.
    # Adding a small constant helps stability during optimization if params go wild.
    # However, 1.0 is already there. If K1, K2 are negative and large, it can vanish.
    # We can use np.where to handle signs, but simple addition is faster.
    # Let's just add a tiny epsilon to prevent exact 0.
    denom_safe = denominator + 1e-9 * np.sign(denominator)
    # Actually, if denom is negative, sign is -1, so we subtract? 
    # Better: just use np.maximum(denom, 1e-6) if we assume denom should be positive?
    # Physical model implies denom > 0 for eps > 0.
    # For eps < 0, denom = 1 - |K1|*|eps| - |K2|*eps^2.
    # Given eps is very small negative, 1 dominates.
    
    hardening = numerator / denom_safe
    
    return E * eps + hardening
