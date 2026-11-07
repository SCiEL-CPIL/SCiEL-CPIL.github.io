"""Nonlinear oscillator symbolic regression using a physics-informed basis."""
# EVOLVE-BLOCK-START
import numpy as np

# Normalization constants derived from training ranges
X_MIN = -0.5728
X_MAX = 0.5275
V_MIN = -0.1754
V_MAX = 0.2821

X_MID = (X_MIN + X_MAX) / 2.0
X_HALF = (X_MAX - X_MIN) / 2.0
V_MID = (V_MIN + V_MAX) / 2.0
V_HALF = (V_MAX - V_MIN) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for a nonlinear oscillator.
    Uses a physics-motivated basis:
    - Linear restoring force (p0 * x)
    - Cubic nonlinearity (p1 * x^3)
    - Damping (p2 * v)
    - Quadratic damping (p3 * v^2)
    - Cross terms (p4 * x*v, p5 * x^2*v, p6 * x*v^2)
    - Higher order (p7 * x^5, p8 * x^2*v^2)
    - Bias (p9)
    """
    # Normalize inputs to [-1, 1] range to improve conditioning
    xn = (x - X_MID) / X_HALF
    vn = (v - V_MID) / V_HALF
    
    # Construct basis matrix (10 columns)
    # 1. x (linear restoring)
    # 2. x^3 (cubic nonlinearity)
    # 3. v (damping)
    # 4. v^2 (quadratic damping)
    # 5. x*v (cross term)
    # 6. x^2*v (cross term)
    # 7. x*v^2 (cross term)
    # 8. x^5 (higher order restoring)
    # 9. x^2*v^2 (higher order cross)
    # 10. bias
    
    basis = np.column_stack((
        xn,
        xn**3,
        vn,
        vn**2,
        xn * vn,
        xn**2 * vn,
        xn * vn**2,
        xn**5,
        xn**2 * vn**2,
        np.ones_like(x)
    ))
    
    # Compute prediction
    return basis @ params
# EVOLVE-BLOCK-END
