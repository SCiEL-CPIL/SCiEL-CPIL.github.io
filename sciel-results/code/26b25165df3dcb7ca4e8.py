# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with normalized inputs.
    
    Normalization is applied to x, v, and t to ensure all basis functions
    are of similar magnitude, improving the conditioning of the optimization
    problem for BFGS.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions (on normalized inputs):
    p0: Constant
    p1: Linear position (x_norm)
    p2: Linear velocity (v_norm)
    p3: Quadratic position (x_norm^2)
    p4: Cubic position (x_norm^3) - Duffing term
    p5: Quadratic velocity (v_norm^2)
    p6: Cubic velocity (v_norm^3) - Nonlinear damping
    p7: Cross term (x_norm * v_norm)
    p8: Sinusoidal forcing (sin(t_norm))
    p9: Cosinusoidal forcing (cos(t_norm))
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly unit scale
    # Ranges: t [30, 50] -> center 40, scale 10
    # x [-0.2, 0.17] -> scale 0.2 (approx max abs)
    # v [-0.34, 0.36] -> scale 0.35 (approx max abs)
    
    t_norm = (t - 40.0) / 10.0
    x_norm = x / 0.2
    v_norm = v / 0.35
    
    # Construct basis functions
    f0 = np.ones_like(x_norm)
    f1 = x_norm
    f2 = v_norm
    f3 = x_norm ** 2
    f4 = x_norm ** 3
    f5 = v_norm ** 2
    f6 = v_norm ** 3
    f7 = x_norm * v_norm
    f8 = np.sin(t_norm)
    f9 = np.cos(t_norm)
    
    # Linear combination of basis functions
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
