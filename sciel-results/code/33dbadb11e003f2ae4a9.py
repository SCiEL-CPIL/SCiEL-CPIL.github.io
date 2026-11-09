"""Chebyshev polynomial basis for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev-based model for nonlinear oscillator acceleration.
    
    Uses Chebyshev polynomials in normalized x and v for well-conditioned
    approximation of the nonlinear acceleration field, plus sinusoidal drive terms.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Normalize x and v to [-1, 1] for Chebyshev evaluation
    # x range: [-0.2056, 0.1749] -> center = -0.01535, half_range = 0.19025
    # v range: [-0.3391, 0.3624] -> center = 0.01165, half_range = 0.35075
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    xn = (x - x_center) / x_half
    vn = (v - v_center) / v_half
    
    # Clip to [-1, 1] to keep Chebyshev well-defined
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials T0, T1, T2, T3
    T0_x = np.ones_like(xn)
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    
    T0_v = np.ones_like(vn)
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn
    
    # Basis: 2 drive terms + 8 Chebyshev terms in (x, v)
    # Drive: sin(t), cos(t)
    # State: T0_x, T1_x, T2_x, T3_x, T0_v, T1_v, T2_v, T3_v
    # That's 10 total
    
    b = np.column_stack((
        np.sin(t),      # 0: sine drive
        np.cos(t),      # 1: cosine drive
        T0_x,           # 2: Chebyshev T0(x) = 1
        T1_x,           # 3: Chebyshev T1(x) = x_norm
        T2_x,           # 4: Chebyshev T2(x)
        T3_x,           # 5: Chebyshev T3(x)
        T0_v,           # 6: Chebyshev T0(v) = 1 (redundant with T0_x but ok)
        T1_v,           # 7: Chebyshev T1(v) = v_norm
        T2_v,           # 8: Chebyshev T2(v)
        T3_v            # 9: Chebyshev T3(v)
    ))
    
    # Linear combination
    result = b @ p
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
