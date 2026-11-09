import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a scaled monomial basis on the normalized input.
    Map x from [0.1, 6] to t in [0, 1], then use powers t^0..t^9.
    This provides a simple, smooth basis that BFGS can optimize effectively
    when starting from all-ones, avoiding the correlation issues of
    higher-order orthogonal polynomials.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x to [0, 1]
    # x_min = 0.1, x_max = 6.0
    t = (x - 0.1) / 5.9
    
    # Build output as sum of params[i] * t^i
    y = np.zeros_like(x)
    power = np.ones_like(x)
    
    for i in range(10):
        y += params[i] * power
        power = power * t
    
    return y
