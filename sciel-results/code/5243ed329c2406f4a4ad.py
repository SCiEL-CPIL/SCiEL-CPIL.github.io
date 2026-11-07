"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a rich basis of common functional forms.
    The basis is designed to span polynomials, trigonometric, and exponential
    functions, allowing BFGS to fit a wide variety of smooth targets.
    
    params: array of length 10
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure finite input
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    # Build basis matrix: each column is a basis function evaluated at x
    # We have 10 parameters, so 10 basis functions
    
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic
    b2 = x ** 2
    
    # 4. Cubic
    b3 = x ** 3
    
    # 5. Quartic
    b4 = x ** 4
    
    # 6. Sine
    b5 = np.sin(x)
    
    # 7. Cosine
    b6 = np.cos(x)
    
    # 8. Tanh
    b7 = np.tanh(x)
    
    # 9. Exponential (scaled to avoid overflow/underflow issues)
    # Use exp(x/2) which is well-behaved on [-2, 2]
    b8 = np.exp(0.5 * x)
    
    # 10. Sine with higher frequency
    b9 = np.sin(2.0 * x)
    
    # Combine: prediction = sum(params[i] * b_i)
    # Use vectorized computation
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure output is finite
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
