# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a rational function structure.
    
    Structure:
    f(x, y, z) = (p0 + p1*x + p2*y + p3*z + p4/x + p5/y + p6/z) 
                 / (1 + p7*x + p8*y + p9*z)
    
    This form allows for:
    1. Rational dependencies and saturation effects.
    2. Reciprocal (pole-like) behavior in the numerator.
    3. Linear interactions in the denominator.
    
    The +1.0 in the denominator ensures numerical stability for inputs in [0.5, 2]
    and parameters near 1, preventing division by zero or near-zero values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    if params.size < 10:
        params = np.concatenate([params, np.zeros(10 - params.size)])
    
    # Extract 10 parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params[:10]
    
    # Safe reciprocals (inputs are in [0.5, 2], so no division by zero)
    x_safe = np.where(np.abs(x) > 1e-12, x, 1e-12)
    y_safe = np.where(np.abs(y) > 1e-12, y, 1e-12)
    z_safe = np.where(np.abs(z) > 1e-12, z, 1e-12)
    
    # Build numerator: constant, linear, and reciprocal terms
    numerator = (
        p0 +
        p1 * x +
        p2 * y +
        p3 * z +
        p4 / x_safe +
        p5 / y_safe +
        p6 / z_safe
    )
    
    # Build denominator: 1 + linear terms
    denominator = (
        1.0 +
        p7 * x +
        p8 * y +
        p9 * z
    )
    
    # Safe division
    # Clip denominator to prevent division by zero or near-zero values
    # If denominator is very small, we replace it with a small value with the same sign
    # But since inputs are positive and params start at 1, denominator is likely positive.
    # We use a robust clipping strategy.
    abs_den = np.abs(denominator)
    sign_den = np.sign(denominator)
    # If sign is 0 (denominator is exactly 0), we use 1.0 to keep it positive
    sign_den = np.where(sign_den == 0, 1.0, sign_den)
    
    # Clip denominator magnitude to be at least 1e-6
    den_safe = np.where(abs_den < 1e-6, 1e-6 * sign_den, denominator)
    
    pred = numerator / den_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
