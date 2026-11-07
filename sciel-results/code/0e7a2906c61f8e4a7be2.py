import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a separable product basis:
    f(x,y,z) ≈ (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z)
    
    This uses all 10 parameters to model multiplicative interactions
    and exponential-like growth that global linear/additive models cannot capture.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters for each factor
    a0, a1, a2 = p[0], p[1], p[2]
    b0, b1 = p[3], p[4]
    c0, c1 = p[5], p[6]
    
    # Additional parameters for cross terms or scaling
    d0, d1, d2, d3, d4 = p[7], p[8], p[9], 0.0, 0.0
    
    # Compute each factor
    fx = a0 + a1 * x + a2 * x * x
    fy = b0 + b1 * y
    fz = c0 + c1 * z
    
    # Product of the three factors
    result = fx * fy * fz
    
    # Add linear correction terms using remaining parameters
    # This helps capture non-separable components
    result = result + d0 * x + d1 * y + d2 * z
    
    return result
