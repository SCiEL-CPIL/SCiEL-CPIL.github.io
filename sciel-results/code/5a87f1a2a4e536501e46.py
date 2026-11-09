import numpy as np

# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

# Scale to roughly [-1, 1] for better conditioning
x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator.
    
    Basis terms (10 total):
    0: 1 (bias)
    1: xn (linear in x)
    2: xn^3 (Duffing cubic restoring)
    3: vn (linear in v)
    4: xn^2 * vn (van der Pol nonlinear damping)
    5: vn^3 (cubic damping)
    6: xn * vn^2 (mixed interaction)
    7: xn * vn (bilinear interaction)
    8: vn^2 (quadratic damping)
    9: xn^2 * vn^2 (higher-order interaction)
    
    Inputs normalized to [-1, 1] for conditioning.
    """
    # Normalize inputs
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale

    # Precompute powers
    x2 = xn * xn
    x3 = x2 * xn
    v2 = vn * vn
    v3 = v2 * vn

    # Basis terms
    t0 = np.ones_like(xn)
    t1 = xn
    t2 = x3
    t3 = vn
    t4 = x2 * vn
    t5 = v3
    t6 = xn * v2
    t7 = xn * vn
    t8 = v2
    t9 = x2 * v2

    # Linear combination
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

    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)

    return result
