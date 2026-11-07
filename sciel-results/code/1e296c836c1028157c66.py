"""Symbolic regression for E. coli growth rate - Chebyshev polynomial basis of degree 3."""
import numpy as np


def _chebyshev(x, n):
    """Compute Chebyshev polynomial of degree n evaluated at x (vectorized)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        T0 = np.ones_like(x)
        T1 = x.copy()
        for k in range(2, n + 1):
            T2 = 2.0 * x * T1 - T0
            T0, T1 = T1, T2
        return T1


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 3 for E. coli growth rate.
    
    Uses a linear-in-parameters model with Chebyshev basis functions.
    The 10 parameters correspond to:
        p[0]: bias (constant term)
        p[1]: T1(b)
        p[2]: T2(b)
        p[3]: T3(b)
        p[4]: T1(s)
        p[5]: T2(s)
        p[6]: T3(s)
        p[7]: T1(temp)
        p[8]: T2(temp)
        p[9]: T3(temp)
    
    Note: pH is not explicitly included in the 10-parameter budget to allow
    richer representation of the more important variables. If pH is important,
    it could be combined with other terms, but given the 10-parameter limit,
    we focus on b, s, and temp which likely have stronger effects.
    """
    p = params
    
    # Normalize inputs to [-1, 1] for Chebyshev polynomial stability
    # b: [1, 4.998] -> [-1, 1]
    b_norm = (2.0 * b - (1.0 + 4.998)) / (4.998 - 1.0)
    # s: [1, 39.98] -> [-1, 1]
    s_norm = (2.0 * s - (1.0 + 39.98)) / (39.98 - 1.0)
    # temp: [1, 12] -> [-1, 1]
    temp_norm = (2.0 * temp - (1.0 + 12.0)) / (12.0 - 1.0)
    
    # Clip to [-1, 1] for stability
    b_norm = np.clip(b_norm, -1.0, 1.0)
    s_norm = np.clip(s_norm, -1.0, 1.0)
    temp_norm = np.clip(temp_norm, -1.0, 1.0)
    
    # Compute Chebyshev basis functions up to degree 3
    T1_b = _chebyshev(b_norm, 1)
    T2_b = _chebyshev(b_norm, 2)
    T3_b = _chebyshev(b_norm, 3)
    
    T1_s = _chebyshev(s_norm, 1)
    T2_s = _chebyshev(s_norm, 2)
    T3_s = _chebyshev(s_norm, 3)
    
    T1_t = _chebyshev(temp_norm, 1)
    T2_t = _chebyshev(temp_norm, 2)
    T3_t = _chebyshev(temp_norm, 3)
    
    # Linear combination of basis functions
    mu = (p[0] + 
          p[1] * T1_b + p[2] * T2_b + p[3] * T3_b +
          p[4] * T1_s + p[5] * T2_s + p[6] * T3_s +
          p[7] * T1_t + p[8] * T2_t + p[9] * T3_t)
    
    # Clip to reasonable range to avoid extreme values
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu
