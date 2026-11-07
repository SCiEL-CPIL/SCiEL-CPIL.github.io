import numpy as np
import time

def _softplus(x, scale=1.0):
    """Smooth approximation to max(x, 0) using softplus."""
    return scale * np.log1p(np.exp(x / scale))

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Ramberg-Osgood formulation
    with temperature-dependent parameters.
    
    sigma = E(T) * epsilon + K(T) * (epsilon + c(T))^n(T)
    
    Uses smooth parameterizations to avoid non-differentiable points that hurt BFGS.
    
    Parameters (10):
    p0, p1, p2: E(T) = p0 + p1*T + p2*T^2 (elastic modulus)
    p3, p4: K(T) = p3 + p4*T (hardening coefficient)
    p5, p6: c(T) = p5 + p6*T (offset for power law)
    p7, p8: n(T) = p7 + p8*T (hardening exponent, smoothed to stay > 0.1)
    p9: correction factor
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    E = p0 + p1 * T + p2 * T * T
    K = p3 + p4 * T
    c = p5 + p6 * T
    
    # Smoothly constrain n to be > 0.1 using softplus
    # n = 0.1 + softplus(p7 + p8*T) ensures n > 0.1 and is smooth
    n_raw = p7 + p8 * T
    n = 0.1 + _softplus(n_raw, scale=0.5)
    
    # Ramberg-Osgood form with offset
    # Use smooth positive base: base = 1e-8 + softplus(u + c)
    base_raw = u + c
    base = 1e-8 + _softplus(base_raw, scale=0.01)
    
    # Power law term
    power_term = np.power(base, n)
    
    # Combine elastic and plastic contributions
    sigma = E * u + K * power_term
    
    # Apply correction factor
    sigma = sigma * (1.0 + p9 * u * T)
    
    return sigma
