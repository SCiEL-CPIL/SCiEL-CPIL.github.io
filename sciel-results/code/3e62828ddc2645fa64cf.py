import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational stress-strain model for aluminium rod.
    Uses a rational function form that smoothly transitions from elastic
    (linear) to plastic (sub-linear) behavior, with temperature-dependent
    corrections.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters for the rational model
    # E(T): effective elastic modulus
    E = p0 + p1 * temp
    # Yield-like stress scale (nominator saturation level)
    S_y = p2 + p3 * temp
    # Denominator coefficient controlling transition sharpness
    # Larger |a| means sharper transition from linear to plateau
    a = np.abs(p4) + 1e-8
    # Hardening slope after transition
    K = p5 + p6 * temp
    # Offset / baseline
    c0 = p7 + p8 * temp
    
    # Rational form: stress = E * strain * S_y / (S_y + a * strain) + K * strain^2 * (something)
    # Actually, let's use a cleaner form:
    # stress = (E * S_y * strain) / (S_y + a * strain) + K * strain^2
    # At small strain: stress ≈ E * strain (elastic)
    # At large strain: stress ≈ E * S_y / a + K * strain^2 (plateau + hardening)
    
    denom = S_y + a * strain
    # Protect against zero denominator
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    
    stress = (E * S_y * strain) / denom_safe + K * strain**2 + c0
    
    # Add cross-term correction for temperature-strain interaction
    stress += p9 * strain * temp
    
    return stress
