"""Symbolic regression with RBF-like localized basis for 2D inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich nonlinear basis of 10 functions.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Design a basis that spans common functional forms:
    # Linear terms, quadratic terms, interaction, and some nonlinearities
    # to capture smooth non-polynomial behavior.
    
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. x
    b1 = x
    
    # 3. y
    b2 = y
    
    # 4. x^2
    b3 = x ** 2
    
    # 5. y^2
    b4 = y ** 2
    
    # 6. x * y
    b5 = x * y
    
    # 7. sin(x)
    b6 = np.sin(x)
    
    # 8. sin(y)
    b7 = np.sin(y)
    
    # 9. cos(x * y)
    b8 = np.cos(x * y)
    
    # 10. exp(-(x^2 + y^2))  -- Gaussian bump at origin
    b9 = np.exp(-(x ** 2 + y ** 2))
    
    # Combine with parameters
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
