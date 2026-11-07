# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining Chebyshev polynomials (for well-conditioned polynomial
    approximation) with non-polynomial features (sin, cos, tanh, exp) to capture
    diverse function shapes.
    
    Basis (10 functions):
      0: T_0(t) = 1
      1: T_1(t) = t
      2: T_2(t) = 2t^2 - 1
      3: T_3(t) = 4t^3 - 3t
      4: T_4(t) = 8t^4 - 8t^2 + 1
      5: sin(x)
      6: cos(x)
      7: tanh(x)
      8: exp(-x^2)
      9: exp(-|x|)
    
    where t = x / 2.0 (scaling from [-2,2] to [-1,1] for Chebyshev conditioning).
    
    This hybrid approach gives BFGS a rich function space that is both well-conditioned
    (via Chebyshev) and capable of capturing periodic, saturating, and decaying behaviors.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4 using recurrence
    # T_0 = 1, T_1 = t, T_n = 2*t*T_{n-1} - T_{n-2}
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Non-polynomial basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x * x)
    exp_neg_abs_x = np.exp(-np.abs(x))
    
    # Stack basis functions
    basis = np.column_stack([
        T0, T1, T2, T3, T4,
        sin_x, cos_x, tanh_x, exp_neg_x2, exp_neg_abs_x
    ])
    
    # Linear combination with params
    result = basis @ params
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
