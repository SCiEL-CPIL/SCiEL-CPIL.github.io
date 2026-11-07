import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model combining polynomial basis with parametric forcing.
    
    Uses centered/scaled inputs for better conditioning.
    Basis:
    1. xn (linear restoring)
    2. xn^3 (cubic restoring / Duffing)
    3. vn (linear damping)
    4. vn^3 (cubic damping)
    5. xn*vn (cross coupling)
    6. sin(t) (forcing)
    7. cos(t) (forcing phase)
    8. xn*cos(t) (parametric modulation)
    9. xn*sin(t) (parametric modulation)
    10. 1 (bias)
    """
    # Deadline guard
    t0 = time.time()
    
    # Center and scale inputs for conditioning
    # x range: [-0.2056, 0.1749], center ~ -0.015, scale ~ 0.2
    # v range: [-0.3391, 0.3624], center ~ 0.01, scale ~ 0.35
    x_mean = -0.015
    x_std = 0.2
    v_mean = 0.01
    v_std = 0.35
    
    xn = (x - x_mean) / x_std
    vn = (v - v_mean) / v_std
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Time terms
    st = np.sin(t)
    ct = np.cos(t)
    
    # Basis functions
    b1 = xn
    b2 = xn3
    b3 = vn
    b4 = vn3
    b5 = xn * vn
    b6 = st
    b7 = ct
    b8 = xn * ct
    b9 = xn * st
    b10 = np.ones_like(x)
    
    # Combine with parameters
    pred = (
        params[0] * b1 +
        params[1] * b2 +
        params[2] * b3 +
        params[3] * b4 +
        params[4] * b5 +
        params[5] * b6 +
        params[6] * b7 +
        params[7] * b8 +
        params[8] * b9 +
        params[9] * b10
    )
    
    return pred

# EVOLVE-BLOCK-END
