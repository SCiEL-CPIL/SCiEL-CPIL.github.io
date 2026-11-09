import numpy as np

# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator (Duffing-van der Pol hybrid).
    
    Basis terms:
    0: 1 (bias)
    1: x_n (linear restoring)
    2: x_n^3 (Duffing nonlinearity)
    3: v_n (linear damping)
    4: x_n^2 * v_n (Van der Pol damping)
    5: v_n^3 (cubic damping)
    6: x_n * v_n^2 (interaction)
    7: x_n^2 (quadratic restoring)
    8: v_n^2 (quadratic damping)
    9: x_n * v_n (interaction)
    
    Inputs are normalized to [-1, 1] range approximately to improve conditioning.
    """
    # Normalize inputs
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale

    # Compute basis terms
    # Using explicit multiplications for speed and clarity
    x2 = xn * xn
    x3 = x2 * xn
    v2 = vn * vn
    v3 = v2 * vn
    
    t0 = np.ones_like(xn)
    t1 = xn
    t2 = x3
    t3 = vn
    t4 = x2 * vn
    t5 = v3
    t6 = xn * v2
    t7 = x2
    t8 = v2
    t9 = xn * vn

    # Linear combination of basis terms with fitted parameters
    result = (params[0] * t0 +
              params[1] * t1 +
              params[2] * t2 +
              params[3] * t3 +
              params[4] * t4 +
              params[5] * t5 +
              params[6] * t6 +
              params[7] * t7 +
              params[8] * t8 +
              params[9] * t9)

    # Ensure finite values to prevent optimizer failure
    result = np.where(np.isfinite(result), result, 0.0)

    return result
