"""Symbolic regression: 10-parameter RBF + polynomial hybrid basis for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis for 2D function recovery using a hybrid of
    Gaussian RBFs (for local flexibility) and low-order polynomials
    (for global structure).
    
    The 10 parameters are fitted by BFGS from the evaluator.
    
    Features:
      0: 1 (constant)
      1: x
      2: y
      3: x^2 - y^2  (quadratic harmonic)
      4: 2*x*y      (quadratic interaction)
      5: RBF at (0, 0) with width 0.6
      6: RBF at (0.5, 0.5) with width 0.5
      7: RBF at (-0.5, -0.5) with width 0.5
      8: RBF at (0.5, -0.5) with width 0.5
      9: RBF at (-0.5, 0.5) with width 0.5
    """
    p = np.asarray(params, dtype=float)
    
    # Polynomial part
    f0 = np.ones_like(x, dtype=float)
    f1 = x
    f2 = y
    f3 = x * x - y * y
    f4 = 2.0 * x * y
    
    # RBF part: Gaussian kernels centered at strategic points
    # Width chosen to balance local flexibility with global smoothness
    w1 = 0.6
    w2 = 0.5
    
    # RBF at origin
    r2_0 = x * x + y * y
    f5 = np.exp(-r2_0 / (w1 * w1))
    
    # RBF at (0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5
    r2_1 = dx * dx + dy * dy
    f6 = np.exp(-r2_1 / (w2 * w2))
    
    # RBF at (-0.5, -0.5)
    dx = x + 0.5
    dy = y + 0.5
    r2_2 = dx * dx + dy * dy
    f7 = np.exp(-r2_2 / (w2 * w2))
    
    # RBF at (0.5, -0.5)
    dx = x - 0.5
    dy = y + 0.5
    r2_3 = dx * dx + dy * dy
    f8 = np.exp(-r2_3 / (w2 * w2))
    
    # RBF at (-0.5, 0.5)
    dx = x + 0.5
    dy = y - 0.5
    r2_4 = dx * dx + dy * dy
    f9 = np.exp(-r2_4 / (w2 * w2))
    
    pred = (
        p[0] * f0 +
        p[1] * f1 +
        p[2] * f2 +
        p[3] * f3 +
        p[4] * f4 +
        p[5] * f5 +
        p[6] * f6 +
        p[7] * f7 +
        p[8] * f8 +
        p[9] * f9
    )
    
    return pred
# EVOLVE-BLOCK-END
