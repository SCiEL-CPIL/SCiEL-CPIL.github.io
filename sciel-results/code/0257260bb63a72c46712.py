"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Structural Change: Autonomous Full 3rd-Order Polynomial in x and v.
    
    Hypothesis: The system is an autonomous nonlinear oscillator (e.g., Duffing or 
    Van der Pol type) where acceleration depends only on position (x) and velocity (v), 
    not explicitly on time (t). The previous model included time-dependent terms 
    (linear drift and sinusoidal drive) which may have caused overfitting or 
    misalignment with the true dynamics.
    
    This model uses a complete polynomial basis of degree 3 in x and v:
    1.  1
    2.  x
    3.  v
    4.  x^2
    5.  x*v
    6.  v^2
    7.  x^3
    8.  x^2*v
    9.  x*v^2
    10. v^3
    
    The inputs are scaled to be O(1) to improve the condition number of the 
    design matrix for the BFGS optimizer.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale inputs to approximate unit magnitude for numerical stability
    # x range approx [-0.2, 0.17] -> scale by 0.2
    # v range approx [-0.34, 0.36] -> scale by 0.35
    x_s = x * 5.0   # 1/0.2
    v_s = v * 2.857 # 1/0.35

    # Compute powers
    x2 = x_s * x_s
    x3 = x2 * x_s
    v2 = v_s * v_s
    v3 = v2 * v_s

    # Basis functions (Complete polynomial degree 3 in 2 variables)
    b0 = np.ones_like(x)      # 1
    b1 = x_s                  # x
    b2 = v_s                  # v
    b3 = x2                   # x^2
    b4 = x_s * v_s            # x*v
    b5 = v2                   # v^2
    b6 = x3                   # x^3
    b7 = x2 * v_s             # x^2*v
    b8 = x_s * v2             # x*v^2
    b9 = v3                   # v^3

    # Linear combination
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)

    return pred
# EVOLVE-BLOCK-END
