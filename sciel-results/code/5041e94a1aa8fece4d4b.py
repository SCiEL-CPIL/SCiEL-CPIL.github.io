"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid rational-tanh model with temperature-dependent coefficients.
    
    Model:
    sigma = Rational_Term + Tanh_Term + Quad_Correction
    
    Rational_Term = E(T) * eps / (1 + K(T) * eps)
    E(T) = p0 + p1 * T + p2 * T^2
    K(T) = p3 + p4 * T
    
    Tanh_Term = Y(T) * tanh(R(T) * eps)
    Y(T) = p5 + p6 * T
    R(T) = p7 + p8 * T
    
    Quad_Correction = p9 * eps^2
    
    Parameters (10 constants):
    p0, p1, p2: Elastic modulus / Scale coefficients (quadratic in T)
    p3, p4: Saturation rate coefficients (linear in T)
    p5, p6: Tanh amplitude coefficients (linear in T)
    p7, p8: Tanh rate coefficients (linear in T)
    p9: Quadratic strain correction coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    K_T = p3 + p4 * T
    Y_T = p5 + p6 * T
    R_T = p7 + p8 * T
    
    # Rational term: captures elastic region and saturation
    # Safe division: if denominator is close to 0, clip it
    den = 1.0 + K_T * eps
    # Ensure denominator doesn't cause instability
    den_safe = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6, den)
    rational_term = (E_T * eps) / den_safe
    
    # Tanh term: captures yield point and hardening behavior
    tanh_term = Y_T * np.tanh(R_T * eps)
    
    # Quadratic correction
    quad_term = p9 * eps * eps
    
    sigma = rational_term + tanh_term + quad_term
    
    return sigma
# EVOLVE-BLOCK-END
