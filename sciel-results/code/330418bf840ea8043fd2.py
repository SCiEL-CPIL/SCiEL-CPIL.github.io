"""Rewritten program for symbolic regression with a rational and interaction basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a basis of linear, reciprocal, and interaction terms.
    
    The basis is chosen to capture functions that may involve:
    - Linear dependencies (x, y, z)
    - Hyperbolic or decaying behaviors (1/x, 1/y, 1/z)
    - Pairwise multiplicative interactions (xy, yz, zx)
    
    This design addresses the limitations of pure polynomial or pure exponential bases
    by allowing the model to fit rational-like structures and simple non-linear interactions
    which are common in symbolic regression benchmarks.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Construct the feature matrix (n_samples, 10)
    # Basis: 1, x, y, z, 1/x, 1/y, 1/z, xy, yz, zx
    # Using np.where to avoid division by zero, though inputs are [0.5, 2]
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        inv_x,
        inv_y,
        inv_z,
        x * y,
        y * z,
        z * x
    ))
    
    # Compute prediction: features (n, 10) dot params (10,) -> (n,)
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
