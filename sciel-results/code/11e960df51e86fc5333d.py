"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich, well-conditioned basis of 10 terms.
    
    The basis combines:
    - Chebyshev polynomials T_0 to T_4 (polynomial-like, well-conditioned on [-1,1])
    - Sine and cosine terms at different frequencies (for periodic components)
    - A linear term (captured by T_1)
    
    This mixture covers both polynomial and periodic structure.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for stability
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    T2 = 2.0 * t_safe * T1 - T0
    T3 = 2.0 * t_safe * T2 - T1
    T4 = 2.0 * t_safe * T3 - T2
    
    # Trigonometric terms (using x directly, since domain is [-3,3])
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    sin2 = np.sin(2.0 * x)
    cos2 = np.cos(2.0 * x)
    
    # Build the linear combination
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * sin1 +
              params[6] * cos1 +
              params[7] * sin2 +
              params[8] * cos2 +
              params[9] * np.ones_like(t_safe))
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
