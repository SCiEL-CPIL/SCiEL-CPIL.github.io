import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Elastic + yield saturation + plastic hardening with smooth onset.
    sigma = E(T)*eps + A(T)*tanh(B(T)*eps) + C(T)*eps^2*(1-exp(-D(T)*eps)) + F(T)*eps^3*(1-exp(-G(T)*eps))
    
    The (1-exp(-k*eps)) terms create smooth onset for hardening terms,
    capturing the elastic-to-plastic transition. Temperature dependence
    is linear in T for all coefficients.
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
    
    # Temperature-dependent coefficients
    E_T = E0 * (1.0 + c_E * T)
    A_T = A0 + A1 * T
    B_T = B0 + B1 * T
    C_T = C0 + C1 * T
    D_T = D0 + D1 * T
    
    # Base terms
    linear_term = E_T * eps
    tanh_term = A_T * np.tanh(B_T * eps)
    
    # Smooth onset for quadratic hardening: C*eps^2*(1-exp(-D*eps))
    # For numerical stability, clip the exponent argument
    arg = D_T * eps
    arg = np.clip(arg, -50, 50)
    smooth_quad = 1.0 - np.exp(-arg)
    quad_term = C_T * eps * eps * smooth_quad
    
    # We only have 10 params, so we need to be strategic.
    # Let me use a simpler but effective model:
    # sigma = E*eps + A*tanh(B*eps) + C*eps^2 + D*eps^3
    # but with better temperature dependence
    
    # Actually, let me reconsider. The 10 params are:
    # E0, c_E, A0, A1, B0, B1, C0, C1, D0, D1
    # That's exactly what we have. Let me use a clean polynomial + tanh model.
    
    # Reset to a well-structured model
    stress = linear_term + tanh_term + C_T * eps * eps + D_T * eps * eps * eps
    
    return stress
