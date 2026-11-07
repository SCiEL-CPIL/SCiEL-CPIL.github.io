"""Symbolic regression with 2 inputs using a hybrid polynomial + RBF basis + BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial terms for global trends + RBF terms for localized features.
    
    The 10 parameters are:
      p[0]  : constant (1)
      p[1]  : x
      p[2]  : y
      p[3]  : x^2
      p[4]  : y^2
      p[5]  : x*y
      p[6]  : RBF centered at (0, 0) with fixed width
      p[7]  : RBF centered at (0.5, 0.5) with fixed width
      p[8]  : RBF centered at (-0.5, -0.5) with fixed width
      p[9]  : RBF centered at (0.5, -0.5) with fixed width
    
    The RBF centers are fixed at strategic points in [-1, 1]^2 to capture
    localized features, while the polynomial terms handle global structure.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Polynomial terms (6 terms)
    term_0 = np.ones_like(x)          # constant
    term_1 = x                         # x
    term_2 = y                         # y
    term_3 = x ** 2                    # x^2
    term_4 = y ** 2                    # y^2
    term_5 = x * y                     # x*y
    
    # RBF terms (4 terms) with fixed centers and widths
    # Width sigma = 0.5 gives reasonable localization on [-1, 1]^2
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma  # 2*sigma^2 for Gaussian kernel
    
    # RBF centered at (0, 0)
    dist0 = x ** 2 + y ** 2
    term_6 = np.exp(-dist0 / sigma2)
    
    # RBF centered at (0.5, 0.5)
    dist1 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    term_7 = np.exp(-dist1 / sigma2)
    
    # RBF centered at (-0.5, -0.5)
    dist2 = (x + 0.5) ** 2 + (y + 0.5) ** 2
    term_8 = np.exp(-dist2 / sigma2)
    
    # RBF centered at (0.5, -0.5)
    dist3 = (x - 0.5) ** 2 + (y + 0.5) ** 2
    term_9 = np.exp(-dist3 / sigma2)
    
    # Stack all 10 terms
    basis = np.column_stack([
        term_0, term_1, term_2, term_3, term_4, term_5,
        term_6, term_7, term_8, term_9
    ])
    
    # Linear model: prediction = basis @ params
    predictions = basis @ params
    
    # Ensure output is finite and correct shape
    if not np.all(np.isfinite(predictions)):
        return np.zeros_like(x)
        
    return predictions


# EVOLVE-BLOCK-END
