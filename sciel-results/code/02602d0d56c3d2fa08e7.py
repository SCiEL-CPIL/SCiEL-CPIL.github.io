import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a power-law hardening model
    with temperature-dependent coefficients and polynomial corrections.
    
    sigma = E(T) * eps + K(T) * |eps|^n * sign(eps) + polynomial corrections
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp          # Elastic modulus, linear in T
    K = p2 + p3 * temp          # Hardening coefficient, linear in T
    n = 1.0 + np.abs(p4)        # Hardening exponent, ensure n >= 1
    
    # Power-law hardening term: K * |eps|^n * sign(eps)
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    # Handle the case where strain is exactly 0
    power_term = np.where(abs_strain > 1e-15, np.power(abs_strain, n), 0.0)
    hardening = K * power_term * sign_strain
    
    # Elastic term
    elastic = E * strain
    
    # Base stress: elastic + hardening
    stress_base = elastic + hardening
    
    # Polynomial corrections to capture systematic deviations
    # These are lower-order terms that can fit residual patterns
    corrections = (p5 * strain**2 + 
                   p6 * temp * strain + 
                   p7 * temp**2 * strain + 
                   p8 * strain * temp**2 + 
                   p9 * strain**3)
    
    stress = stress_base + corrections
    
    return stress
