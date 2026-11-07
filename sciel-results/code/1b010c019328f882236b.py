import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a hyperbolic tangent yield transition.
    
    sigma = E(T) * u + A(T) * tanh(B(T) * (u - u_y(T)))
    
    Where:
      E(T) = p0 + p1*T          (Elastic modulus)
      A(T) = p2 + p3*T          (Yield stress amplitude)
      B(T) = softplus(p4) + softplus(p5)*T  (Sharpness of yield transition)
      u_y(T) = p6 + p7*T        (Yield strain)
      p8, p9: Additional quadratic and cubic correction terms
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Yield stress amplitude coefficients
    p4, p5: Yield transition sharpness coefficients
    p6, p7: Yield strain coefficients
    p8: Quadratic correction
    p9: Cubic correction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Stable softplus for positive parameters
    def stable_softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    # Temperature-dependent coefficients
    E = p0 + p1 * T
    A = p2 + p3 * T
    B_base = stable_softplus(p4)
    B_temp = stable_softplus(p5)
    B = B_base + B_temp * T
    u_y = p6 + p7 * T
    
    # Elastic term
    elastic = E * u
    
    # Yield transition term using tanh
    # This creates a smooth transition from elastic to plastic behavior
    delta_u = u - u_y
    yield_term = A * np.tanh(B * delta_u)
    
    # Polynomial corrections for additional flexibility
    quad = p8 * u * u
    cubic = p9 * u * u * u
    
    # Combine all terms
    sigma = elastic + yield_term + quad + cubic
    
    return sigma
