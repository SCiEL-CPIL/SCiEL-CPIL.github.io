"""Symbolic regression: recover hidden function via diverse basis expansion fit by BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-params basis function expansion with improved conditioning.
    params: array of length 10
    Basis: 1, t, t^2, t^3, t^4, sin(x), cos(x), tanh(x), sin(3x), cos(3x)
    where t = x/3 for improved numerical conditioning of the polynomial part.
    This makes the problem convex in params, so BFGS converges reliably.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).flatten()
    n_points = len(x)
    if n_points == 0:
        return np.array([], dtype=np.float64)
        
    # Extract 10 coefficients
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning of polynomials
    t = x / 3.0
    
    # Polynomial basis (using t for conditioning)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    
    # Trigonometric basis (using original x for natural frequency)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_3x = np.sin(3.0 * x)
    cos_3x = np.cos(3.0 * x)
    
    # Hyperbolic tangent basis
    tanh_x = np.tanh(x)
    
    # Combine all basis functions
    result = (
        c0 +
        c1 * t +
        c2 * t2 +
        c3 * t3 +
        c4 * t4 +
        c5 * sin_x +
        c6 * cos_x +
        c7 * tanh_x +
        c8 * sin_3x +
        c9 * cos_3x
    )
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return a safe fallback if something went wrong
        return np.zeros_like(x, dtype=np.float64)
        
    # Check deadline
    if time.time() - t_start > 140.0:
        # If we're running out of time, still return result (should be fast)
        pass
        
    return result
# EVOLVE-BLOCK-END
