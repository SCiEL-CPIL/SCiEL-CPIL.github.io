"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a symmetric hybrid rational-tanh model with temperature-dependent coefficients.
    Positivity constraints are applied to physical constants (E, K) using exp() to ensure
    stability and physical realism.
    
    Model:
    sigma = Rational_Term + Tanh_Term + Odd_Poly_Correction
    
    Rational_Term = E(T) * eps / (1 + K(T) * |eps|)
    E(T) = exp(p0 + p1 * T + p2 * T^2)  # Positive definite
    K(T) = exp(p3 + p4 * T)             # Positive definite
    
    Tanh_Term = Y(T) * tanh(R(T) * eps)
    Y(T) = p5 + p6 * T
    R(T) = p7 + p8 * T
    
    Odd_Poly_Correction = p9 * eps * |eps|
    
    Parameters (10 constants):
    p0, p1, p2: Log-Elastic modulus coefficients (quadratic in T)
    p3, p4: Log-Saturation rate coefficients (linear in T)
    p5, p6: Tanh amplitude coefficients (linear in T)
    p7, p8: Tanh rate coefficients (linear in T)
    p9: Odd polynomial correction coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    # E(T) and K(T) are forced positive via exp
    # This prevents poles and sign flips in the rational term
    E_T = np.exp(p0 + p1 * T + p2 * T * T)
    K_T = np.exp(p3 + p4 * T)
    
    Y_T = p5 + p6 * T
    R_T = p7 + p8 * T
    
    # Symmetric Rational term
    abs_eps = np.abs(eps)
    den = 1.0 + K_T * abs_eps
    
    # Denominator is always > 1 if K_T > 0 and abs_eps >= 0
    # So no safe division needed, but keep it for numerical safety if exp overflows/underflows
    # exp of large positive is inf. exp of large negative is 0.
    # If K_T is 0, den is 1.
    # If K_T is inf, den is inf. rational term is 0.
    # This is stable.
    
    rational_term = (E_T * eps) / den
    
    # Tanh term
    tanh_term = Y_T * np.tanh(R_T * eps)
    
    # Odd polynomial correction
    quad_term = p9 * eps * abs_eps
    
    sigma = rational_term + tanh_term + quad_term
    
    return sigma
# EVOLVE-BLOCK-END
