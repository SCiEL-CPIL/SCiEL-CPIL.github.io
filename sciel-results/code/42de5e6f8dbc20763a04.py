"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a diverse basis of 10 terms.
    
    The basis includes Chebyshev polynomials for polynomial components,
    trigonometric functions for periodic behavior, tanh for saturation,
    and a Gaussian for localized bumps. This diverse basis allows BFGS
    to fit a wide variety of target functions well.
    
    We use a pure linear combination so that BFGS can find the optimal
    coefficients efficiently (the problem is linear in params).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clip to ensure numerical stability
    t_safe = np.clip(t_safe, -1.0, 1.0)
    
    # Build basis functions
    # Using a mix of polynomial, trig, and other smooth functions
    
    # Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    T2 = 2.0 * t_safe**2 - 1.0
    T3 = 4.0 * t_safe**3 - 3.0 * t_safe
    T4 = 8.0 * t_safe**4 - 8.0 * t_safe**2 + 1.0
    
    # Additional diverse terms
    sin_t = np.sin(np.pi * t_safe)
    cos_t = np.cos(np.pi * t_safe)
    tanh_t = np.tanh(2.0 * t_safe)
    exp_neg_t2 = np.exp(-t_safe**2)
    
    # Higher frequency sinusoid
    sin_2pi = np.sin(2.0 * np.pi * t_safe)
    
    basis = np.stack([
        T0, T1, T2, T3, T4,
        sin_t, cos_t,
        tanh_t, exp_neg_t2,
        sin_2pi
    ], axis=1)  # shape (n, 10)
    
    # Normalize each basis column to unit L2 norm over the sample
    # This improves conditioning for BFGS
    norms = np.linalg.norm(basis, axis=0, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    basis = basis / norms
    
    # Linear combination
    result = basis @ params
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
