"""Rewritten program for symbolic regression with quadratic polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a basis of quadratic polynomial terms.
    
    Basis: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z
    
    This basis can represent any function of the form:
    f(x, y, z) = p0 + p1*x + p2*y + p3*z + p4*x^2 + p5*y^2 + p6*z^2 + p7*x*y + p8*x*z + p9*y*z
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Construct features
    # 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        x * x,
        y * y,
        z * z,
        x * y,
        x * z,
        y * z
    ))
    
    # Compute prediction
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
