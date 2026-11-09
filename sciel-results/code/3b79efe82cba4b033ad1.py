import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid rational-quadratic basis for symbolic regression.
    
    Combines:
    - A full quadratic polynomial basis (capturing curvature and interactions)
    - A rational structure (denominator with linear terms) to model hyperbolic/non-polynomial behavior
    - Reciprocal terms (1/x, 1/y, 1/z) for additional flexibility
    
    The 10 parameters are:
    p0-p5: Numerator coefficients for [1, x, y, z, xy, xz]
    p6-p9: Denominator coefficients for [y, z, 1/x, 1/y]
    
    Structure: (p0 + p1*x + p2*y + p3*z + p4*xy + p5*xz) / (1 + p6*y + p7*z + p8*(1/x) + p9*(1/y))
    
    This allows the model to represent functions like:
    - Pure polynomials (if denominator params -> 0)
    - Hyperbolic functions (if denominator dominates)
    - Exponential-like behaviors
    - Functions with asymptotic behavior
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Reciprocal terms (safe since x, y, z in [0.5, 2])
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Interaction terms
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Numerator: full quadratic-like basis
    num = (p[0] * 1.0 +
           p[1] * x +
           p[2] * y +
           p[3] * z +
           p[4] * xy +
           p[5] * xz)
    
    # Denominator: 1 + linear/rational terms
    den = 1.0 + p[6] * y + p[7] * z + p[8] * inv_x + p[9] * inv_y
    
    # Safe division: avoid division by near-zero values
    # The inputs are in [0.5, 2], so den should be well-behaved for reasonable params
    # But during optimization, params can go wild, so we need protection
    abs_den = np.abs(den)
    mask = abs_den > 1e-8
    
    # For points where den is too small, use a large value to penalize
    den_safe = np.where(mask, den, np.where(den >= 0, 1e-8, -1e-8))
    
    pred = num / den_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
