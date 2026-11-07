"""Rewritten program using a Fourier (trigonometric) basis with 10 terms."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model:
    Map x in [-2, 2] to t in [-pi, pi].
    y = sum_{k=1}^{5} a_k * cos(k*t) + sum_{k=1}^{5} b_k * sin(k*t)
    
    This gives 10 parameters (5 cosine amplitudes + 5 sine amplitudes).
    The Fourier basis is orthogonal on [-pi, pi], well-conditioned for BFGS,
    and can represent smooth, oscillatory, or bounded functions well.
    
    params:
    0-4: a_1..a_5 (cosine amplitudes for k=1..5)
    5-9: b_1..b_5 (sine amplitudes for k=1..5)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Map x from [-2, 2] to t in [-pi, pi]
    t = np.pi * x / 2.0
    
    # Extract parameters
    a = params[0:5]  # cosine amplitudes
    b = params[5:10]  # sine amplitudes
    
    # Compute Fourier series
    # y = sum_{k=1}^{5} a_k * cos(k*t) + sum_{k=1}^{5} b_k * sin(k*t)
    y = np.zeros(n, dtype=float)
    for k in range(1, 6):
        y += a[k-1] * np.cos(k * t)
        y += b[k-1] * np.sin(k * t)
    
    return y
# EVOLVE-BLOCK-END
