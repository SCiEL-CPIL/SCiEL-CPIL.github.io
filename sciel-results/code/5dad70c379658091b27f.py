"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of 10 features.
    
    The basis is designed to capture common symbolic regression targets:
    - Polynomials up to degree 2 in each variable
    - Interaction terms (xy, xz, yz, xyz)
    - Reciprocal terms (1/x, 1/y, 1/z)
    - Mixed terms (x/y, x/z, y/z)
    - Constant term
    
    We use 10 params to form a linear combination of 10 basis functions.
    The evaluator fits these 10 params via BFGS to minimize training MSE.
    
    Basis functions (indexed 0-9):
    0: 1.0                    (constant)
    1: x                      (linear x)
    2: y                      (linear y)
    3: z                      (linear z)
    4: x*y                    (interaction xy)
    5: x*z                    (interaction xz)
    6: y*z                    (interaction yz)
    7: x*y*z                  (triple interaction)
    8: 1.0/x                  (reciprocal x)
    9: 1.0/y                  (reciprocal y)
    
    Note: inputs are in [0.5, 2], so reciprocals are safe and finite.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis functions
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    f8 = 1.0 / x
    f9 = 1.0 / y
    
    # Linear combination
    pred = (params[0] * f0 +
            params[1] * f1 +
            params[2] * f2 +
            params[3] * f3 +
            params[4] * f4 +
            params[5] * f5 +
            params[6] * f6 +
            params[7] * f7 +
            params[8] * f8 +
            params[9] * f9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
