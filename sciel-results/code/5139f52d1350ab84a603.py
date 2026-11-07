import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t0 = time.time()
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # All parameters are LINEAR weights; all centers/slopes are FIXED.
    # This makes the objective convex (linear least squares) so BFGS converges globally.
    
    # Params 0-3: Polynomial coefficients (degree 0-3)
    b0, b1, b2, b3 = params[0], params[1], params[2], params[3]
    
    # Params 4-9: Weights for 6 fixed-center tanh units
    # Fixed centers evenly spaced in [-2, 2], fixed slopes for sharpness
    w1, w2, w3, w4, w5, w6 = params[4], params[5], params[6], params[7], params[8], params[9]
    
    # Fixed centers: 6 points evenly spaced in [-1.5, 1.5]
    # (avoiding edges to allow polynomial to handle boundary behavior)
    centers = np.array([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
    # Fixed slopes: moderate sharpness to capture local features
    slopes = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    
    # Compute polynomial part using Horner's method
    poly = b0 + x * (b1 + x * (b2 + x * b3))
    
    # Compute tanh part with fixed centers and slopes
    # tanh(s * (x - c)) for each unit, weighted by learned weights
    # Vectorized for speed
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    tanh_terms = np.tanh(slopes[np.newaxis, :] * diff)
    tanh_part = tanh_terms @ np.array([w1, w2, w3, w4, w5, w6])
    
    result = poly + tanh_part
    
    # Time guard: if we're getting close to the limit, return what we have
    if time.time() - t0 > 140:
        return result
    
    return result
