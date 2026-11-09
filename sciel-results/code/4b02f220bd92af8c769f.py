import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 9th-degree Chebyshev polynomial expansion.
    
    Args:
        x: Input array of shape (N,) with values in [-3, 3].
        params: Array of 10 parameters to be fitted.
        
    Returns:
        Prediction array of shape (N,).
    """
    # Normalize input to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.empty((10, x.size), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    
    for i in range(2, 10):
        T[i] = 2.0 * t * T[i-1] - T[i-2]
        
    # Linear combination of Chebyshev polynomials
    # params has shape (10,), T has shape (10, N)
    # We want output of shape (N,)
    y = np.dot(params, T)
    
    # Ensure finite output to avoid penalty during fitting
    # If params become huge, this might overflow, but BFGS usually handles scaling.
    # A simple check is safe.
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
