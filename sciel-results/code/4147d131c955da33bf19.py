import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials T_0..T_5 for global smoothness,
    plus x*tanh(x) for asymmetric smooth saturation, plus sin/cos for
    periodic components. This covers a wider function family than pure
    Chebyshev or pure polynomial+trig.
    """
    # Ensure x is 1D for processing
    original_shape = x.shape
    x_flat = x.flatten()
    N = len(x_flat)
    
    if N == 0:
        return np.empty(original_shape)
    
    # Scale to [-1, 1] for Chebyshev
    t = x_flat / 2.0
    
    # Build basis matrix (N, 10)
    B = np.empty((N, 10), dtype=float)
    
    # Chebyshev polynomials T_0..T_5
    B[:, 0] = 1.0
    B[:, 1] = t
    for k in range(2, 6):
        B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
    
    # x * tanh(x) - captures asymmetric smooth saturation
    x_val = x_flat
    B[:, 6] = x_val * np.tanh(x_val)
    
    # sin and cos for periodic components
    B[:, 7] = np.sin(x_val)
    B[:, 8] = np.cos(x_val)
    
    # x^2 * tanh(x) - captures another type of asymmetric behavior
    B[:, 9] = (x_val * x_val) * np.tanh(x_val)
    
    p = params.flatten()
    y = B @ p
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
