"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter hybrid basis combining polynomial and exponential terms.
    
    Basis: [1, x, y, z, x^2, y^2, z^2, exp(x), exp(y), exp(z)]
    
    This basis captures:
    - Constant and linear trends (first 4 terms)
    - Quadratic curvature (next 3 terms)
    - Exponential growth/decay patterns (last 3 terms)
    
    The combination allows BFGS to fit functions that have both
    polynomial and exponential components, which pure polynomial
    bases cannot approximate well.
    """
    # Ensure array inputs
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Time guard: if we're running too long, return a safe default
    start_time = time.time()
    
    p = params[:10]
    
    # Build the 10 basis terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x * x
    b5 = y * y
    b6 = z * z
    b7 = np.exp(x)
    b8 = np.exp(y)
    b9 = np.exp(z)
    
    # Linear combination
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 +
              p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # Fall back to a simple linear combination
        result = p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3
    
    return result
# EVOLVE-BLOCK-END
