import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev (0-4) + Sinusoidal + Exponential.
    params:
      0-4: Chebyshev coefficients c0..c4
      5: Amplitude A for sin
      6: Raw frequency w (mapped to [-5, 5])
      7: Raw phase phi (mapped to [-pi, pi])
      8: Coefficient B for exp
      9: Raw rate kappa (mapped to [-2, 2])
    """
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Linear combination of Chebyshev terms
    y_cheb = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 + 
              params[4] * T4)
    
    # Sinusoidal term: A * sin(w * x + phi)
    A = params[5]
    w = 5.0 * np.tanh(params[6])
    phi = np.pi * np.tanh(params[7])
    y_sin = A * np.sin(w * x + phi)
    
    # Exponential term: B * exp(kappa * x)
    B = params[8]
    kappa = 2.0 * np.tanh(params[9])
    # Clip exponent to prevent overflow, though tanh limits it to [-2,2] so max exp is exp(6) ~ 400
    y_exp = B * np.exp(kappa * x)
    
    return y_cheb + y_sin + y_exp
