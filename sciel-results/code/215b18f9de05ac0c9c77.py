"""Rewritten program for symbolic regression using a Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a Chebyshev polynomial basis.
    
    The inputs x, y, z are mapped from [0.5, 2] to [-1, 1] to utilize the
    properties of Chebyshev polynomials. The basis consists of 10 terms
    formed by products of 1D Chebyshev polynomials T_0, T_1, T_2 up to
    total degree 2. This provides a stable, orthogonal-like representation
    for smooth functions, capturing linear, quadratic, and interaction effects
    more effectively than raw monomials.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1]
    # mid = 1.25, half_range = 0.75
    # u = (v - 1.25) / 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Clip to [-1, 1] to handle any slight numerical overflow outside the domain
    # though inputs are guaranteed in [0.5, 2]
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Chebyshev polynomials T_0, T_1, T_2
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2.0 * u * u - 1.0
    
    T0_v = np.ones_like(v)
    T1_v = v
    T2_v = 2.0 * v * v - 1.0
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2.0 * w * w - 1.0
    
    # Construct the 10 basis terms
    # 1: T0(u)T0(v)T0(w)
    # 2: T1(u)T0(v)T0(w)
    # 3: T0(u)T1(v)T0(w)
    # 4: T0(u)T0(v)T1(w)
    # 5: T2(u)T0(v)T0(w)
    # 6: T0(u)T2(v)T0(w)
    # 7: T0(u)T0(v)T2(w)
    # 8: T1(u)T1(v)T0(w)
    # 9: T1(u)T0(v)T1(w)
    # 10: T0(u)T1(v)T1(w)
    
    features = np.column_stack((
        T0_u * T0_v * T0_w,
        T1_u * T0_v * T0_w,
        T0_u * T1_v * T0_w,
        T0_u * T0_v * T1_w,
        T2_u * T0_v * T0_w,
        T0_u * T2_v * T0_w,
        T0_u * T0_v * T2_w,
        T1_u * T1_v * T0_w,
        T1_u * T0_v * T1_w,
        T0_u * T1_v * T1_w
    ))
    
    # Compute prediction
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
