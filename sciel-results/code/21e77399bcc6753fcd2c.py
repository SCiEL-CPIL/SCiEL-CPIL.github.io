import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a multiplicative temperature-modulated
    elastic-plastic formulation with hyperbolic tangent hardening.
    
    Structure:
    1. Temperature-dependent elastic modulus E(T).
    2. Temperature-dependent yield stress sigma_y(T).
    3. Temperature-dependent yield strain eps_y(T).
    4. Temperature-dependent hardening rate.
    5. Hardening term uses tanh for smooth S-curve transition.
    
    Parameters:
    p0: Elastic modulus at T=0
    p1: Temperature coefficient for elastic modulus
    p2: Yield stress at T=0
    p3: Temperature coefficient for yield stress
    p4: Base hardening rate
    p5: Base yield strain
    p6: Temperature coefficient for hardening rate
    p7: Temperature coefficient for yield strain
    p8: Intercept
    p9: Quadratic elastic correction coefficient
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * T
    
    # Temperature-dependent yield stress
    sigma_y = p2 + p3 * T
    
    # Temperature-dependent yield strain (multiplicative modulation)
    eps_y = p5 * (1.0 + p7 * T)
    
    # Plastic strain
    eps_p = np.maximum(0.0, eps - eps_y)
    
    # Temperature-dependent hardening rate (multiplicative modulation)
    rate = p4 * (1.0 + p6 * T)
    
    # Hardening term: hyperbolic tangent saturation
    # Smooth S-curve: linear for small eps_p, saturates to 1 for large eps_p
    hardening_stress = sigma_y * np.tanh(rate * eps_p)
    
    # Elastic stress contribution
    elastic_stress = E * eps
    
    # Higher-order elastic correction
    correction = p9 * eps * eps
    
    # Total stress
    stress = elastic_stress + hardening_stress + correction + p8
    
    return stress
