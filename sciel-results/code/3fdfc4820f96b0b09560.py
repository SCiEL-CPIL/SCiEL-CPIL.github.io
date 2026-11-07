"""Symbolic regression: Chebyshev basis + periodic terms for richer function family."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (T0..T4) + 5 periodic features.
    
    The periodic features (sin/cos at two frequencies) allow the model to
    capture oscillatory components that pure polynomials cannot, while the
    Chebyshev terms handle the smooth polynomial-like trend.
    
    Design:
      - T_0, T_1, T_2, T_3, T_4 (degree 0-4) on t in [-1,1]
      - sin(π t), cos(π t), sin(2π t), cos(2π t), sin(3π t)
    
    Total: 10 basis functions, 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Periodic features
    pi_t = np.pi * t
    s1 = np.sin(pi_t)
    c1 = np.cos(pi_t)
    s2 = np.sin(2.0 * pi_t)
    c2 = np.cos(2.0 * pi_t)
    s3 = np.sin(3.0 * pi_t)
    
    # Stack basis (N, 10)
    X = np.column_stack([T0, T1, T2, T3, T4, s1, c1, s2, c2, s3])
    
    return X @ p
# EVOLVE-BLOCK-END
