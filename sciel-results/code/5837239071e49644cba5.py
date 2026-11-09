import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a multiplicative temperature-modulated
    elastic-plastic formulation.
    
    Structure:
    1. Temperature-dependent elastic modulus E(T) as a linear polynomial.
    2. Temperature-dependent yield stress sigma_y(T) as a linear polynomial.
    3. Plastic strain epsilon_p = max(0, eps - eps_y)
    4. Hardening term: sigma_y(T) * (1 - exp(-p4 * epsilon_p))
       This captures exponential saturation of hardening.
    5. Total stress: E(T) * eps + sigma_y(T) * (1 - exp(-p4 * max(0, eps - eps_y)))
    
    Parameters:
    p0: Elastic modulus at T=0
    p1: Temperature coefficient for elastic modulus (E(T) = p0 + p1*T)
    p2: Yield stress at T=0
    p3: Temperature coefficient for yield stress (sigma_y(T) = p2 + p3*T)
    p4: Hardening saturation rate (inverse characteristic plastic strain)
    p5: Yield strain location eps_y
    p6: Additional elastic correction coefficient
    p7: Additional elastic correction temperature coefficient
    p8: Intercept (residual stress offset)
    p9: Higher-order elastic correction
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus: E(T) = p0 + p1*T + p6*eps + p7*eps*T
    # This allows for slight nonlinearity in the elastic region
    E = p0 + p1 * T + p6 * eps + p7 * eps * T
    
    # Elastic stress contribution
    elastic_stress = E * eps
    
    # Temperature-dependent yield stress: sigma_y(T) = p2 + p3*T
    sigma_y = p2 + p3 * T
    
    # Plastic strain (only for eps > yield strain)
    eps_p = np.maximum(0.0, eps - p5)
    
    # Hardening term: exponential saturation
    # For small eps_p, this is approximately sigma_y * p4 * eps_p (linear hardening)
    # For large eps_p, this saturates to sigma_y
    hardening_stress = sigma_y * (1.0 - np.exp(-p4 * eps_p))
    
    # Higher-order correction
    correction = p9 * eps * eps
    
    # Total stress
    stress = elastic_stress + hardening_stress + correction + p8
    
    return stress
