"""Rewritten symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of monomials and interactions.
    The evaluator fits params (length 10) via BFGS starting from ones.
    We use a standardized basis to improve conditioning for the optimizer.
    """
    p = params.flatten()
    
    # Ensure at least 10 params; if not, pad with zeros (should not happen per contract)
    if len(p) < 10:
        p = np.concatenate([p, np.zeros(10 - len(p))])
    
    # Standardize inputs to [0.5, 2] range to center around 0 and scale to ~1
    # Mean = 1.25, Std = sqrt((2-0.5)^2/12) = sqrt(2.25/12) = sqrt(0.1875) ≈ 0.43301
    mean = 1.25
    std = 0.4330127
    
    X = (x - mean) / std
    Y = (y - mean) / std
    Z = (z - mean) / std
    
    # Basis functions:
    # 0: 1 (bias)
    # 1: X
    # 2: Y
    # 3: Z
    # 4: X*Y
    # 5: X*Z
    # 6: Y*Z
    # 7: X^2
    # 8: Y^2
    # 9: Z^2
    
    # This basis spans all quadratic polynomials in x, y, z.
    # It is a structural change from linear to quadratic, allowing the model
    # to capture interactions and curvature.
    
    pred = (p[0] * 1.0 + 
            p[1] * X + 
            p[2] * Y + 
            p[3] * Z + 
            p[4] * X * Y + 
            p[5] * X * Z + 
            p[6] * Y * Z + 
            p[7] * X * X + 
            p[8] * Y * Y + 
            p[9] * Z * Z)
    
    return pred
# EVOLVE-BLOCK-END
