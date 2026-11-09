import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier basis with a linear term.
    Basis: {1, x, sin(k*pi/3*x), cos(k*pi/3*x) for k=1..4}
    This provides a well-conditioned, orthogonal-like basis for smooth functions
    on [-3, 3], which is more robust for BFGS than high-degree polynomials.
    
    Parameters:
    x (np.ndarray): Input array of shape (N,), values in [-3, 3].
    params (np.ndarray): Array of 10 coefficients to be fitted.
    
    Returns:
    np.ndarray: Predictions of shape (N,).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    if len(x) == 0:
        return np.zeros_like(x)
        
    result = np.zeros_like(x)
    
    # p0: Constant
    result += params[0]
    
    # p1: Linear
    result += params[1] * x
    
    # p2-p9: Fourier terms
    # Frequencies: pi/3, 2*pi/3, pi, 4*pi/3
    # This corresponds to periods 6, 3, 2, 1.5
    
    # k=1
    result += params[2] * np.sin(np.pi / 3.0 * x)
    result += params[3] * np.cos(np.pi / 3.0 * x)
    
    # k=2
    result += params[4] * np.sin(2.0 * np.pi / 3.0 * x)
    result += params[5] * np.cos(2.0 * np.pi / 3.0 * x)
    
    # k=3
    result += params[6] * np.sin(np.pi * x)
    result += params[7] * np.cos(np.pi * x)
    
    # k=4
    result += params[8] * np.sin(4.0 * np.pi / 3.0 * x)
    result += params[9] * np.cos(4.0 * np.pi / 3.0 * x)
    
    # Check deadline (unlikely to be needed for simple vector ops)
    if time.time() > deadline:
        # Return what we have (which is all of it)
        pass
        
    return result
