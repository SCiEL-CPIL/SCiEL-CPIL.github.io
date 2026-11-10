import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 9th-degree Chebyshev polynomial expansion
    with scaled basis functions to improve BFGS convergence from the all-ones start.
    
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
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # Use decaying scale for higher-order terms to improve BFGS convergence
    # from the all-ones initialization. This reduces the effective magnitude
    # of higher-order parameters, making the loss landscape smoother.
    scale = np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05])
    
    y = (params[0] * scale[0] * T0 +
         params[1] * scale[1] * T1 +
         params[2] * scale[2] * T2 +
         params[3] * scale[3] * T3 +
         params[4] * scale[4] * T4 +
         params[5] * scale[5] * T5 +
         params[6] * scale[6] * T6 +
         params[7] * scale[7] * T7 +
         params[8] * scale[8] * T8 +
         params[9] * scale[9] * T9)
    
    # Ensure finite output to avoid penalty during fitting
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
