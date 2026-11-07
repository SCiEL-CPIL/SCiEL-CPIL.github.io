"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled Quadratic Polynomial Basis.
    
    The previous programs suffered from poor optimization performance (scores > 1.0)
    likely due to ill-conditioning of the feature matrix when using raw inputs in [0.5, 2].
    By scaling the inputs to [-1, 1], the basis functions are better conditioned,
    allowing BFGS to converge to a better local minimum.
    
    This basis spans the space of quadratic functions, which is a superset of linear
    functions. If the target is linear, this will fit perfectly. If non-linear,
    it will capture the curvature.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale inputs to [-1, 1] for better numerical conditioning
    # Input range is [0.5, 2.0], center is 1.25, half-range is 0.75
    x_s = (x - 1.25) / 0.75
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    # 10 basis functions (Quadratic)
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = y_s
    b3 = z_s
    b4 = x_s * x_s
    b5 = y_s * y_s
    b6 = z_s * z_s
    b7 = x_s * y_s
    b8 = x_s * z_s
    b9 = y_s * z_s
    
    # Compute prediction
    result = (p[0] * b0 +
              p[1] * b1 +
              p[2] * b2 +
              p[3] * b3 +
              p[4] * b4 +
              p[5] * b5 +
              p[6] * b6 +
              p[7] * b7 +
              p[8] * b8 +
              p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
