import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a hybrid trigonometric-polynomial basis.
    
    This basis is designed to capture common nonlinear dynamics:
    - Linear and cubic terms for restoring force and damping.
    - Quadratic terms for asymmetry or squared effects.
    - Trigonometric terms for periodic nonlinearities.
    - Interaction term for coupling.
    
    Inputs are scaled to improve numerical conditioning for BFGS optimization.
    """
    # Scaling factors based on training ranges
    # x range: [-0.5728, 0.5275] -> scale by approx 0.55
    # v range: [-0.1754, 0.2821] -> scale by approx 0.25
    x_scale = 0.55
    v_scale = 0.25
    
    X = x / x_scale
    V = v / v_scale
    
    # Build the 10-term basis
    # 1. Bias
    t0 = np.ones_like(X)
    
    # Position terms
    t1 = X
    t2 = X * X
    t3 = X * X * X
    t4 = np.sin(X)
    t5 = np.cos(X)
    
    # Velocity terms
    t6 = V
    t7 = V * V
    t8 = V * V * V
    
    # Interaction term
    t9 = X * V
    
    # Combine into a matrix
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    # Compute prediction
    return T @ params
