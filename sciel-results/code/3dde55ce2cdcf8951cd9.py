"""
Symbolic regression: product-of-sums structure with multiplicative interactions.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, used in a product-of-sums structure:
        (p0 + p1*x + p2*y) * (p3 + p4*x + p5*y) + p6*x*y + p7*x^2*y + p8*y^2*x + p9
    
    This captures multiplicative interactions that linear combinations approximate poorly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Product of two linear forms
    left = p0 + p1 * x + p2 * y
    right = p3 + p4 * x + p5 * y
    product_term = left * right
    
    # Additional interaction terms for flexibility
    interaction = p6 * x * y + p7 * x**2 * y + p8 * y**2 * x
    
    pred = product_term + interaction + p9
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
