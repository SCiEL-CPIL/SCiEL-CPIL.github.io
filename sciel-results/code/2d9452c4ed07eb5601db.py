"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    Uses normalized inputs to improve BFGS conditioning.
    Assumes autonomous dynamics (Van der Pol / Duffing type).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs based on training ranges to unit scale
    # x range ~ [-0.34, 0.36], v range ~ [-0.21, 0.17]
    # Using 0.4 and 0.2 as scaling factors for robustness
    xn = x / 0.4
    vn = v / 0.2

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Basis terms
    # 1. Constant
    t1 = np.ones_like(x)
    # 2. Linear position
    t2 = xn
    # 3. Linear velocity
    t3 = vn
    # 4. Quadratic position
    t4 = xn2
    # 5. Cross x*v
    t5 = xn * vn
    # 6. Quadratic velocity
    t6 = vn2
    # 7. Cubic position
    t7 = xn3
    # 8. Van der Pol style damping (x^2 * v)
    t8 = xn2 * vn
    # 9. Mixed x * v^2
    t9 = xn * vn2
    # 10. Cubic velocity
    t10 = vn3

    return (p0 * t1 + p1 * t2 + p2 * t3 + p3 * t4 + p4 * t5 +
            p5 * t6 + p6 * t7 + p7 * t8 + p8 * t9 + p9 * t10)
# EVOLVE-BLOCK-END
