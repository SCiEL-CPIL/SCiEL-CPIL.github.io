"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Use a well-conditioned basis that BFGS can optimize effectively
    # Mix of low-order polynomials and trigonometric functions
    # Avoid high-order polynomials that cause ill-conditioning
    
    f0 = np.ones_like(t)
    f1 = t
    f2 = t * t
    f3 = np.sin(np.pi * t)
    f4 = np.cos(np.pi * t)
    f5 = np.sin(2 * np.pi * t)
    f6 = np.cos(2 * np.pi * t)
    f7 = t**3
    f8 = np.tanh(2 * t)
    f9 = t**2 * np.sin(np.pi * t)
    
    # Stack features into a matrix of shape (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    return features.dot(params)

# EVOLVE-BLOCK-END
