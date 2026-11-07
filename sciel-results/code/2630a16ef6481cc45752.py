"""Symbolic regression for a nonlinear oscillator using a physics-informed basis."""
# EVOLVE-BLOCK-START
import numpy as np

# Normalization constants derived from training ranges
# x: [-0.5728, 0.5275]
# v: [-0.1754, 0.2821]
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
    Uses a physics-motivated monomial basis on normalized inputs:
    - Restoring force: x, x^3, x^5
    - Damping: v, v^2, v^3
    - Cross terms: xv, x^2v, xv^2
    - Bias
    """
    # Normalize inputs to [-1, 1] range to improve conditioning
    # This helps BFGS converge as features are O(1)
    xn = (x - X_MID) / X_HALF
    vn = (v - V_MID) / V_HALF
    
    # Precompute powers for efficiency
    xn2 = xn * xn
    xn3 = xn2 * xn
    xn5 = xn3 * xn2
    
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Construct basis matrix (10 columns)
    # 1. Bias
    # 2. Linear x (Restoring)
    # 3. Cubic x (Nonlinear Restoring)
    # 4. Quintic x (Higher Order Restoring)
    # 5. Linear v (Damping)
    # 6. Quadratic v (Drag)
    # 7. Cubic v (Higher Order Drag)
    # 8. x*v (Coupling)
    # 9. x^2*v (Velocity-dependent Stiffness)
    # 10. x*v^2 (Position-dependent Drag)
    
    basis = np.column_stack((
        np.ones_like(xn),
        xn,
        xn3,
        xn5,
        vn,
        vn2,
        vn3,
        xn * vn,
        xn2 * vn,
        xn * vn2
    ))
    
    # Compute prediction
    pred = basis @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(xn)
        
    return pred
# EVOLVE-BLOCK-END
