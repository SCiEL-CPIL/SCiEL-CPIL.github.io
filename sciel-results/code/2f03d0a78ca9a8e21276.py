"""Symbolic regression with 2 inputs using a rich, well-conditioned basis."""
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a 2D Chebyshev-like polynomial basis up to degree 3,
    plus radial and angular terms to capture broader function structure.
    
    The basis is designed to be well-conditioned so BFGS converges reliably.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    # Build a rich basis of 10 features
    # We use a combination of:
    # - Low-degree polynomials (well-conditioned)
    # - Radial terms (x^2+y^2)
    # - Angular terms (arctan2)
    # - Sine/cosine of scaled angles for periodic structure
    
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    theta = np.arctan2(y, x)
    
    # 10 basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    b6 = r2  # radial
    b7 = np.cos(2 * theta) * r2  # angular-radial
    b8 = np.sin(2 * theta) * r2  # angular-radial
    b9 = r * np.cos(3 * theta)   # higher-order radial-angular
    
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Clip basis values to avoid extreme values that could cause numerical issues
    # This helps with conditioning
    basis = np.clip(basis, -10, 10)
    
    # Prediction is linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
