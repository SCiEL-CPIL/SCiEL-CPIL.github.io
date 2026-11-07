"""
Symbolic regression: Enhanced hybrid basis with radial, polynomial, and periodic terms.
This basis is designed to capture a wider range of functional forms including
radial symmetries, saddle shapes, and periodic behaviors, which are common in
symbolic regression benchmarks.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        1, x, y, x^2+y^2, x^2-y^2, xy, tanh(x), tanh(y), sin(x^2+y^2), cos(x^2+y^2)
    
    This basis includes:
    - Linear terms (x, y)
    - Radial quadratic terms (x^2+y^2, x^2-y^2)
    - Interaction term (xy)
    - Saturating non-linearities (tanh(x), tanh(y))
    - Radial periodic terms (sin(x^2+y^2), cos(x^2+y^2))
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x**2
    y2 = y**2
    r2 = x2 + y2
    
    # Compute predictions
    pred = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * y +
        p3 * r2 +
        p4 * (x2 - y2) +
        p5 * (x * y) +
        p6 * np.tanh(x) +
        p7 * np.tanh(y) +
        p8 * np.sin(r2) +
        p9 * np.cos(r2)
    )
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
