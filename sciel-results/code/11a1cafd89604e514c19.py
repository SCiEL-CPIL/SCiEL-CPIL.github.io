import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-scale rational stress-strain model for aluminium rod.
    
    Uses a sum of two rational saturation terms to independently model
    the elastic-to-yield transition and the subsequent hardening regime,
    plus a low-order polynomial correction for residual systematic error.
    
    Structure:
      stress = A(T)*s / (B(T) + C(T)*s) + D(T)*s / (E(T) + F(T)*s) + G(T)*s + H(T)*s^2 + I(T)*s^3 + J(T)
    
    Each rational term captures a distinct nonlinear saturation scale.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # First rational term: captures elastic-to-yield transition
    # A1(T) = p0 + p1*temp  (numerator scale)
    # B1(T) = p2 + p3*temp  (denominator offset - yield-like)
    # C1 = p4               (denominator strain coefficient - transition sharpness)
    A1 = p0 + p1 * temp
    B1 = p2 + p3 * temp
    C1 = p4
    
    # Second rational term: captures hardening regime
    # A2(T) = p5 + p6*temp  (numerator scale)
    # B2(T) = p7 + p8*temp  (denominator offset)
    # C2 = p9               (denominator strain coefficient)
    A2 = p5 + p6 * temp
    B2 = p7 + p8 * temp
    C2 = p9
    
    # Polynomial correction terms (using remaining capacity via interactions)
    # We've used all 10 params in the two rational terms, so add
    # a small cubic term derived from existing parameters to add flexibility:
    # Actually, let's restructure to keep it clean:
    # Use params more efficiently:
    # Term 1: (p0 + p1*T) * s / ((p2 + p3*T) + p4*s)
    # Term 2: (p5 + p6*T) * s / ((p7 + p8*T) + p9*s)
    # This uses all 10 params. The sum of two rationals is more expressive
    # than one rational + polynomial, as each can saturate at different scales.
    
    denom1 = B1 + C1 * strain
    denom2 = B2 + C2 * strain
    
    # Protect against zero denominators
    denom1_safe = np.where(np.abs(denom1) < 1e-12, 1e-12, denom1)
    denom2_safe = np.where(np.abs(denom2) < 1e-12, 1e-12, denom2)
    
    term1 = A1 * strain / denom1_safe
    term2 = A2 * strain / denom2_safe
    
    stress = term1 + term2
    
    return stress
