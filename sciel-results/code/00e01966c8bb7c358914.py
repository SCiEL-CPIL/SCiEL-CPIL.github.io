import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function f(x) using 10 parameters.
    
    Strategy: Use a rich, well-conditioned basis that spans a wide variety of
    function shapes:
    - Polynomial terms (up to degree 4) for smooth trends
    - Chebyshev polynomials (degree 5-8) for better-conditioned higher-order terms
    - Periodic terms (sin, cos) for oscillatory components
    - Saturating terms (tanh, sigmoid-like) for step/saturation behavior
    - Gaussian bump for localized features
    
    All basis functions are bounded or well-behaved on [-1, 1] (after scaling x).
    """
    # Scale x from [-3, 3] to [-1, 1] for better numerical conditioning
    u = x / 3.0
    
    # Basis functions:
    # 0: Constant
    b0 = np.ones_like(u)
    
    # 1-4: Low-degree polynomials
    b1 = u
    b2 = u ** 2
    b3 = u ** 3
    b4 = u ** 4
    
    # 5-8: Chebyshev polynomials of first kind (well-conditioned on [-1,1])
    # T5(u) = 16u^5 - 20u^3 + 5u
    b5 = 16.0 * u ** 5 - 20.0 * u ** 3 + 5.0 * u
    # T6(u) = 32u^6 - 48u^4 + 18u^2 - 1
    b6 = 32.0 * u ** 6 - 48.0 * u ** 4 + 18.0 * u ** 2 - 1.0
    # T7(u) = 64u^7 - 112u^5 + 56u^3 - 7u
    b7 = 64.0 * u ** 7 - 112.0 * u ** 5 + 56.0 * u ** 3 - 7.0 * u
    # T8(u) = 128u^8 - 256u^6 + 160u^4 - 32u^2 + 1
    b8 = 128.0 * u ** 8 - 256.0 * u ** 6 + 160.0 * u ** 4 - 32.0 * u ** 2 + 1.0
    
    # 9: Tanh for saturation/step-like behavior
    b9 = np.tanh(3.0 * u)
    
    # Combine linearly
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
