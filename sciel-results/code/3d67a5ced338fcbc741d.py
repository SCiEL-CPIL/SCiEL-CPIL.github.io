"""
Symbolic regression: Rational function with quadratic numerator and denominator.
This basis captures smooth, non-polynomial behaviors including saturation,
pole-like structures, and rapid transitions that linear or pure polynomial
bases cannot represent efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        p0, p1, p2, p3, p4, p5 (numerator coefficients)
        p6, p7, p8, p9 (denominator coefficients, with implicit 1 offset)
    
    The model is:
        (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y) /
        (1 + p6*x + p7*y + p8*x^2 + p9*y^2)
    
    This rational form can approximate:
    - Linear functions (when denominator ~ 1)
    - Saturating functions (when denominator grows)
    - Functions with local features or poles
    - Smooth non-polynomial behaviors
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x**2
    y2 = y**2
    
    # Compute numerator
    numerator = p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * (x * y)
    
    # Compute denominator with implicit 1 offset for stability
    denominator = 1.0 + p6 * x + p7 * y + p8 * x2 + p9 * y2
    
    # Guard against near-zero or negative denominator to avoid numerical issues
    # Use a small epsilon to prevent division by zero
    eps = 1e-8
    denom_safe = np.where(np.abs(denominator) > eps, denominator, np.sign(denominator) * eps)
    # If denominator is exactly zero, use eps with positive sign
    denom_safe = np.where(denominator == 0, eps, denom_safe)
    
    # Compute predictions
    pred = numerator / denom_safe
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback: clip to reasonable range
        pred = np.clip(pred, -1e6, 1e6)
        # If still not finite, use a simple linear fallback
        if not np.all(np.isfinite(pred)):
            pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
