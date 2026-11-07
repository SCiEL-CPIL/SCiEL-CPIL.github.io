import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a separable 2D Chebyshev basis.
    
    The model assumes additive forces:
    y = f(x) + g(v)
    where f and g are Chebyshev polynomial expansions.
    
    This structure:
    1. Enforces physical separability (stiffness depends on x, damping on v).
    2. Reduces parameter correlation compared to full 2D monomial/Chebyshev bases.
    3. Uses Chebyshev polynomials for superior conditioning on bounded domains.
    4. Allocates 5 coefficients to x and 5 to v, allowing up to 4th-order terms
       in each dimension, which is sufficient for smooth oscillator dynamics.
    
    Inputs are mapped to [-1, 1] for the Chebyshev basis.
    """
    # Map inputs to [-1, 1] based on training ranges
    # x range: [-0.5728, 0.5275] -> center = -0.02265, half-width = 0.55015
    # v range: [-0.1754, 0.2821] -> center = 0.05335, half-width = 0.22875
    
    x_center = -0.02265
    x_half = 0.55015
    v_center = 0.05335
    v_half = 0.22875
    
    X = (x - x_center) / x_half
    V = (v - v_center) / v_half
    
    # Clip to [-1, 1] for stability, though training data is in range
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T0 to T4
    # T0 = 1
    # T1 = t
    # T2 = 2*t^2 - 1
    # T3 = 4*t^3 - 3*t
    # T4 = 8*t^4 - 8*t^2 + 1
    
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    
    # Chebyshev terms for X
    Tx0 = np.ones_like(X)
    Tx1 = X
    Tx2 = 2.0 * X2 - 1.0
    Tx3 = 4.0 * X3 - 3.0 * X
    Tx4 = 8.0 * X4 - 8.0 * X2 + 1.0
    
    # Chebyshev terms for V
    Tv0 = np.ones_like(V)
    Tv1 = V
    Tv2 = 2.0 * V2 - 1.0
    Tv3 = 4.0 * V3 - 3.0 * V
    Tv4 = 8.0 * V4 - 8.0 * V2 + 1.0
    
    # Combine: 5 terms for X, 5 terms for V
    # params[0:5] for X, params[5:10] for V
    result = (params[0] * Tx0 +
              params[1] * Tx1 +
              params[2] * Tx2 +
              params[3] * Tx3 +
              params[4] * Tx4 +
              params[5] * Tv0 +
              params[6] * Tv1 +
              params[7] * Tv2 +
              params[8] * Tv3 +
              params[9] * Tv4)
    
    return result
