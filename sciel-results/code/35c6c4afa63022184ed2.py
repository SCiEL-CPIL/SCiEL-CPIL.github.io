"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_chebyshev


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover a hidden function using a combination of basis functions.
    The basis includes Chebyshev polynomials (mapped to [0.1, 6]) and some
    transcendental functions that are common in symbolic regression tasks.
    
    We use 10 parameters.
    params[0..6]: Coefficients for Chebyshev polynomials T_0..T_6 mapped to x in [0.1, 6].
    params[7]: Coefficient for sin(x)
    params[8]: Coefficient for exp(-x)
    params[9]: Coefficient for log(x)
    
    This hybrid basis covers polynomials (via Chebyshev) and common non-linearities.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev stability
    x_min = 0.1
    x_max = 6.0
    # t = 2*(x - x_min)/(x_max - x_min) - 1
    t = (2.0 * (x - x_min)) / (x_max - x_min) - 1.0
    
    # Clip t to [-1, 1] to avoid numerical issues in Chebyshev evaluation
    # though the domain is exactly [0.1, 6], floating point might slightly exceed
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_6
    # T_0 = 1
    # T_1 = t
    # T_2 = 2*t^2 - 1
    # etc.
    
    # We can compute them iteratively or use scipy.special.eval_chebyshev
    # For efficiency and stability, let's compute manually up to degree 6
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    
    # Transcendental basis functions
    # sin(x)
    S = np.sin(x)
    # exp(-x)
    E = np.exp(-x)
    # log(x)
    # x is in [0.1, 6], so log(x) is safe
    L = np.log(x)
    
    # Combine
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * T6 +
         params[7] * S +
         params[8] * E +
         params[9] * L)
    
    return y
# EVOLVE-BLOCK-END
