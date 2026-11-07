"""Rewritten program for symbolic regression with reciprocal and interaction basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a basis of linear, interaction, and reciprocal terms.
    
    Basis: 1, x, y, z, x*y, x*z, y*z, 1/x, 1/y, 1/z
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Construct features
    # 1, x, y, z, xy, xz, yz, 1/x, 1/y, 1/z
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        x * y,
        x * z,
        y * z,
        inv_x,
        inv_y,
        inv_z
    ))
    
    # Compute prediction
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
