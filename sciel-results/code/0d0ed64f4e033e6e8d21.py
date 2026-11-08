"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator from position and velocity.
    Uses a well-conditioned basis with scaled inputs, odd powers, tanh saturation,
    and interaction terms to capture nonlinear restoring, damping, and coupling.
    """
    p = params

    # Scale inputs to O(1) for better-conditioned optimization
    # x in [-0.5728, 0.5275] -> scale by 1.5 gives ~[-0.86, 0.79]
    # v in [-0.1754, 0.2821] -> scale by 4.0 gives ~[-0.70, 1.13]
    x_s = x * 1.5
    v_s = v * 4.0

    # Core powers of scaled position
    x2 = x_s * x_s
    x3 = x2 * x_s
    x5 = x3 * x2

    # Core powers of scaled velocity
    v2 = v_s * v_s
    v3 = v2 * v_s

    # Interaction terms
    xv = x_s * v_s

    # Smooth saturating functions
    tanh_x = np.tanh(x_s)
    tanh_v = np.tanh(v_s)

    # Sinusoidal term for periodic nonlinearity
    sin_x = np.sin(x_s)

    # Linear combination of 10 basis functions
    # 0: x_s        (linear restoring)
    # 1: x_s^3      (cubic restoring)
    # 2: x_s^5      (quintic restoring)
    # 3: v_s        (linear damping)
    # 4: v_s^3      (cubic damping)
    # 5: x_s*v_s    (parametric coupling)
    # 6: tanh(x_s)  (softening stiffness)
    # 7: tanh(v_s)  (softening damping)
    # 8: sin(x_s)   (smooth periodic stiffness)
    # 9: 1          (bias)

    y = (p[0] * x_s +
         p[1] * x3 +
         p[2] * x5 +
         p[3] * v_s +
         p[4] * v3 +
         p[5] * xv +
         p[6] * tanh_x +
         p[7] * tanh_v +
         p[8] * sin_x +
         p[9])

    return y
# EVOLVE-BLOCK-END
