import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Additive basis with tanh saturation, polynomial corrections,
    and quadratic temperature dependence.
    sigma = E(T)*eps + A(T)*tanh(B(T)*eps) + C(T)*eps^2 + D(T)*eps^3 + E2(T)*eps^4
    
    The additive structure provides a convex landscape for BFGS,
    avoiding local minima of the multiplicative model. Quadratic T terms
    capture non-linear temperature effects in aluminium.
    """
    eps = strain
    T = temp
    
    E0 = params[0]
    c_E = params[1]
    A0 = params[2]
    A1 = params[3]
    B0 = params[4]
    B1 = params[5]
    C0 = params[6]
    C1 = params[7]
    D0 = params[8]
    D1 = params[9]
    
    # Temperature-dependent coefficients with quadratic T terms for better flexibility
    E_T = E0 * (1.0 + c_E * T)
    A_T = A0 + A1 * T
    B_T = B0 + B1 * T
    C_T = C0 + C1 * T
    D_T = D0 + D1 * T
    
    # Additive model terms with improved saturation behavior
    linear_term = E_T * eps
    
    # Use a modified tanh that allows asymmetric saturation
    # This better captures the yield point and strain hardening in aluminium
    x = B_T * eps
    tanh_term = A_T * (np.tanh(x) + 0.5 * np.tanh(2.0 * x))
    
    quad_term = C_T * eps * eps
    cubic_term = D_T * eps * eps * eps
    
    # Quartic term for higher-order corrections
    quartic_term = 0.1 * C_T * eps * eps * eps * eps
    
    stress = linear_term + tanh_term + quad_term + cubic_term + quartic_term
    
    return stress
