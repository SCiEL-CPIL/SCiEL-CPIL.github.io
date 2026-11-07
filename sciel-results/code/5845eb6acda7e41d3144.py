import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a rational function.
    
    Structure:
    - Rational function: (a0 + a1*eps + a2*eps^2) / (1 + b0*eps + b1*eps^2)
    - Coefficients are linear in temperature: a_i = p_i + p_{i+5}*T, b_j = p_{j+3} + p_{j+8}*T
    - 10 params: 
      p0, p1, p2: Num base coeffs
      p3, p4: Den base coeffs (excluding 1)
      p5, p6, p7: Num temp coeffs
      p8, p9: Den temp coeffs
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Effective coefficients
    a0 = p0 + p5 * temp
    a1 = p1 + p6 * temp
    a2 = p2 + p7 * temp
    
    b0 = p3 + p8 * temp
    b1 = p4 + p9 * temp
    
    # Rational function
    numerator = a0 + a1 * strain + a2 * strain**2
    denominator = 1.0 + b0 * strain + b1 * strain**2
    
    # Avoid division by zero (safety margin)
    # Given ranges, denominator should be positive if params are reasonable.
    # Add a small epsilon to prevent inf/nan if BFGS wanders.
    denom_safe = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
    
    result = numerator / denom_safe
    
    return result
