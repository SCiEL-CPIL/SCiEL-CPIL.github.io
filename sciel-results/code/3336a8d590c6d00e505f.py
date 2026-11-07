import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    Uses a smooth transition between elastic and plastic regions.
    """
    p = params
    
    # Extract parameters
    p0, p1 = p[0], p[1]  # E(T) = p0 + p1*T
    p2, p3 = p[2], p[3]  # eps_y(T) = p2 + p3*T
    p4, p5 = p[4], p[5]  # H(T) = p4 + p5*T (linear hardening)
    p6, p7 = p[6], p[7]  # Q(T) = p6 + p7*T (quadratic hardening)
    p8 = p[8]            # Sharpness of transition
    p9 = p[9]            # Offset
    
    T = temp
    eps = strain
    
    # Calculate temperature-dependent properties
    E = p0 + p1 * T
    eps_y = p2 + p3 * T
    H = p4 + p5 * T
    Q = p6 + p7 * T
    
    # Elastic stress
    sigma_el = E * eps
    
    # Plastic stress
    # Starts from the elastic stress at the yield point to ensure continuity
    u = eps - eps_y
    sigma_pl = E * eps_y + H * u + Q * u * u
    
    # Smooth transition weight
    # alpha goes from 0 (elastic) to 1 (plastic)
    # Using tanh for smoothness
    # To avoid overflow in tanh for very large p8*u, we can clip or rely on numpy's stability
    # np.tanh is stable for large inputs (returns +/- 1)
    alpha = 0.5 * (1.0 + np.tanh(p8 * u))
    
    # Blend
    sigma = (1.0 - alpha) * sigma_el + alpha * sigma_pl + p9
    
    return sigma
