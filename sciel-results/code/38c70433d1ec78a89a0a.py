"""Symbolic regression with hybrid Chebyshev + Fourier basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 4 Chebyshev polynomials (T0-T3) + 6 Fourier terms (sin/cos at freq 1,2,3).
    Total: 10 parameters.
    
    The Chebyshev part captures global polynomial trends.
    The Fourier part captures periodic oscillations.
    Together they cover a broader class of smooth functions than either alone.
    
    Inputs x are in [0.1, 6]. We map to u in [0, 1] for the Fourier basis,
    and to t in [-1, 1] for the Chebyshev basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Map x from [0.1, 6] to u in [0, 1]
    u = (x - 0.1) / 5.9
    u = np.clip(u, 0.0, 1.0)
    
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # Chebyshev polynomials T0, T1, T2, T3
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    
    # Fourier terms: sin(2*pi*k*u), cos(2*pi*k*u) for k=1,2,3
    # This gives 6 terms
    sin1 = np.sin(2.0 * np.pi * u)
    cos1 = np.cos(2.0 * np.pi * u)
    sin2 = np.sin(4.0 * np.pi * u)
    cos2 = np.cos(4.0 * np.pi * u)
    sin3 = np.sin(6.0 * np.pi * u)
    cos3 = np.cos(6.0 * np.pi * u)
    
    # Stack all 10 basis functions
    X = np.column_stack([
        T0, T1, T2, T3,
        sin1, cos1,
        sin2, cos2,
        sin3, cos3
    ])
    
    # Prediction
    return X @ p
# EVOLVE-BLOCK-END
