"""Rewritten program for symbolic regression using a reciprocal and interaction basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a basis of constant, linear, reciprocal, 
    and pairwise interaction terms.
    
    The basis consists of 10 terms:
    1. 1
    2. x
    3. y
    4. z
    5. 1/x
    6. 1/y
    7. 1/z
    8. x * y
    9. x * z
    10. y * z
    
    This basis is chosen to capture rational (decaying) and multiplicative 
    interaction effects, which are common in symbolic regression targets.
    The inputs are in [0.5, 2], so reciprocals are well-behaved.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Calculate basis terms
    # Ensure no division by zero (inputs are >= 0.5, so this is safe, but good practice)
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Pairwise interactions
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Construct feature matrix (N, 10)
    features = np.column_stack((
        np.ones_like(x),
        x,
        y,
        z,
        inv_x,
        inv_y,
        inv_z,
        xy,
        xz,
        yz
    ))
    
    # Compute prediction
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
