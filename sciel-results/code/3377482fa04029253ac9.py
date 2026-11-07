# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled Quadratic Polynomial Basis with numerical safeguards.
    
    The target function is unknown but the inputs are in [0.5, 2].
    We use a centered/scaled quadratic basis which is well-conditioned
    and spans the space of quadratic polynomials in 3 variables (10 coefficients).
    
    If the true function is quadratic, this recovers it exactly.
    If it's linear or constant, the higher-order terms will be fit to ~0.
    """
    # Deadline guard: return immediately if we're near the limit
    # (though this function is called many times, each call is fast)
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params is exactly length 10
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # Scale inputs to [-1, 1] for better numerical conditioning
    # Input range is [0.5, 2.0], center is 1.25, half-range is 0.75
    x_s = (x - 1.25) / 0.75
    y_s = (y - 1.25) / 0.75
    z_s = (z - 1.25) / 0.75
    
    # 10 basis functions (Quadratic)
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = y_s
    b3 = z_s
    b4 = x_s * x_s
    b5 = y_s * y_s
    b6 = z_s * z_s
    b7 = x_s * y_s
    b8 = x_s * z_s
    b9 = y_s * z_s
    
    # Compute prediction using dot product for efficiency
    # Stack basis functions into a matrix and multiply by params
    # This is equivalent to the sum but more efficient for large arrays
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    result = B @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Replace non-finite values with 0 to avoid invalid results
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
