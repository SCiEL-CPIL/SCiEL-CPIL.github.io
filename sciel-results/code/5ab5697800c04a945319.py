"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a scaled polynomial basis to improve conditioning for BFGS optimization.
    The inputs x and v are scaled to the range [-1, 1] based on their training ranges.
    This prevents higher-order terms from becoming numerically insignificant.
    """
    # Training ranges provided in task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale x and v to [-1, 1]
    x_center = (x_min + x_max) / 2.0
    x_scale = (x_max - x_min) / 2.0
    v_center = (v_min + v_max) / 2.0
    v_scale = (v_max - v_min) / 2.0
    
    # Avoid division by zero (though scales are non-zero here)
    if x_scale == 0: x_scale = 1.0
    if v_scale == 0: v_scale = 1.0

    xs = (x - x_center) / x_scale
    vs = (v - v_center) / v_scale

    # Basis functions using scaled variables
    # Physically motivated terms for nonlinear oscillators
    t1 = xs
    t2 = xs**3
    t3 = xs**2
    t4 = vs
    t5 = vs**3
    t6 = vs**2
    t7 = xs * vs
    t8 = xs * vs**2
    t9 = xs**2 * vs
    t10 = 1.0

    # Combine with fitted parameters
    return (params[0] * t1 +
            params[1] * t2 +
            params[2] * t3 +
            params[3] * t4 +
            params[4] * t5 +
            params[5] * t6 +
            params[6] * t7 +
            params[7] * t8 +
            params[8] * t9 +
            params[9] * t10)
# EVOLVE-BLOCK-END
