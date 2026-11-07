import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis expansion for stress-strain of an aluminium rod.
    
    sigma = p0*phi0 + p1*phi1 + ... + p9*phi9
    
    Basis functions are chosen to capture:
    - Linear elastic response: strain, strain*T, strain*T^2
    - Nonlinear hardening: strain*|strain|^1.5, strain*|strain|^1.5*T
    - Temperature effects: T, T^2, T*strain
    - Constant offset
    
    Being linear in parameters makes the BFGS optimization convex,
    ensuring convergence to the global least-squares minimum.
    """
    p = params
    
    # Ensure inputs are float arrays
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    # If parameters are not finite, return zeros to avoid propagation of NaNs/Infs
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    # Basis functions - linear in parameters
    # Elastic regime (linear in strain)
    phi0 = strain                          # basic elastic
    phi1 = strain * temp                   # elastic temp dependence
    phi2 = strain * temp**2                # quadratic temp dependence
    
    # Hardening (nonlinear in strain, linear in params)
    abs_strain = np.abs(strain)
    phi3 = strain * np.power(abs_strain + 1e-8, 1.5)           # power-law hardening
    phi4 = strain * np.power(abs_strain + 1e-8, 1.5) * temp    # hardening temp dependence
    phi5 = strain * np.power(abs_strain + 1e-8, 2.0)           # higher-order hardening
    
    # Temperature-only terms
    phi6 = temp                            # thermal offset
    phi7 = temp**2                         # quadratic thermal
    
    # Cross terms
    phi8 = temp * np.power(abs_strain + 1e-8, 1.0)             # temp-strain interaction
    
    # Constant
    phi9 = np.ones_like(strain)
    
    sigma = (p[0] * phi0 + p[1] * phi1 + p[2] * phi2 +
             p[3] * phi3 + p[4] * phi4 + p[5] * phi5 +
             p[6] * phi6 + p[7] * phi7 +
             p[8] * phi8 + p[9] * phi9)
    
    return sigma
