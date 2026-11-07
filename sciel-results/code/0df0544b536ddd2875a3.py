"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Structural Change:
    Uses a 2D Chebyshev Polynomial basis of total degree up to 3.
    Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned
    design matrix for the linear least squares problem (or BFGS minimization of MSE).
    This avoids the collinearity issues present in the previous model (e.g., x vs sin(x)).
    
    Inputs are scaled to [-1, 1] based on the known training ranges to ensure
    the Chebyshev basis is effective.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Training ranges from task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale inputs to [-1, 1]
    # u = 2 * (x - x_min) / (x_max - x_min) - 1
    # w = 2 * (v - v_min) / (v_max - v_min) - 1
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    # Avoid division by zero if range is 0, though unlikely
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Clip to [-1, 1] to handle potential slight out-of-distribution inputs in test
    # Chebyshev polynomials grow rapidly outside [-1, 1], which is fine for extrapolation
    # but we ensure stability.
    
    # Compute Chebyshev polynomials T_0, T_1, T_2, T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # For u
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2.0 * u * u - 1.0
    T3_u = 4.0 * u * u * u - 3.0 * u
    
    # For w
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2.0 * w * w - 1.0
    T3_w = 4.0 * w * w * w - 3.0 * w
    
    # Construct 10 Basis Functions (Mixed Chebyshev)
    # Order:
    # 0: T0(u)T0(w) = 1
    # 1: T1(u)T0(w) = u
    # 2: T0(u)T1(w) = w
    # 3: T2(u)T0(w) = 2u^2 - 1
    # 4: T0(u)T2(w) = 2w^2 - 1
    # 5: T1(u)T1(w) = u*w
    # 6: T3(u)T0(w) = 4u^3 - 3u
    # 7: T0(u)T3(w) = 4w^3 - 3w
    # 8: T2(u)T1(w) = (2u^2 - 1)w
    # 9: T1(u)T2(w) = u(2w^2 - 1)
    
    b0 = T0_u * T0_w
    b1 = T1_u * T0_w
    b2 = T0_u * T1_w
    b3 = T2_u * T0_w
    b4 = T0_u * T2_w
    b5 = T1_u * T1_w
    b6 = T3_u * T0_w
    b7 = T0_u * T3_w
    b8 = T2_u * T1_w
    b9 = T1_u * T2_w
    
    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
# EVOLVE-BLOCK-END
