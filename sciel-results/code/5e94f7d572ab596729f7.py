import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis of Chebyshev polynomials 
    and non-polynomial terms.
    
    Structural Change:
    - Replaces raw powers (x, x^2, ...) with Chebyshev polynomials T_0..T_4.
      This improves the condition number of the design matrix, making BFGS 
      more stable and likely to find the global minimum for the linear weights.
    - Retains non-polynomial terms (sin, cos, exp, tanh) but scales them 
      appropriately for the domain [-2, 2].
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Chebyshev Polynomials of the first kind
    # T_0(t) = 1
    b0 = np.ones_like(t)
    # T_1(t) = t
    b1 = t
    # T_2(t) = 2t^2 - 1
    b2 = 2.0 * t**2 - 1.0
    # T_3(t) = 4t^3 - 3t
    b3 = 4.0 * t**3 - 3.0 * t
    # T_4(t) = 8t^4 - 8t^2 + 1
    b4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Non-polynomial terms
    # Scaled to be active on [-2, 2]
    # sin(pi*x/2) goes from 0 to 0, peak at 1.
    b5 = np.sin(np.pi * x / 2.0)
    # cos(pi*x/2) goes from 0 to 0, peak at 1.
    b6 = np.cos(np.pi * x / 2.0)
    
    # Gaussian terms, width ~ 1.0
    x2 = x * x
    b7 = np.exp(-0.5 * x2)
    b8 = x * np.exp(-0.5 * x2)
    
    # Tanh, scaled to saturate around +/- 2
    b9 = np.tanh(x)
    
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
         
    return y
