"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: if we've been running too long, return a simple safe result
    # But since this is called by the optimizer, we need to be fast.
    # The deadline guard is more relevant if we were doing internal optimization,
    # but here we just evaluate the expression.
    
    # Ensure x is a numpy array for consistent operations
    x = np.asarray(x, dtype=np.float64)
    
    # Use a rich basis of functions that can approximate many common target functions.
    # The parameters are 10 in total. We'll use a combination of polynomial, trigonometric,
    # and other common basis functions.
    
    # Polynomial terms (up to degree 4)
    p0 = np.ones_like(x)
    p1 = x
    p2 = x ** 2
    p3 = x ** 3
    p4 = x ** 4
    
    # Trigonometric terms
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(2 * x)
    t4 = np.cos(2 * x)
    
    # Additional useful terms
    # tanh for saturation behavior
    a1 = np.tanh(x)
    # Rational-like term (avoid division by zero)
    a2 = 1.0 / (1.0 + x ** 2)
    
    # Combine with parameters
    # We have 10 parameters, so we pick 10 basis functions
    result = (params[0] * p0 +
              params[1] * p1 +
              params[2] * p2 +
              params[3] * p3 +
              params[4] * p4 +
              params[5] * t1 +
              params[6] * t2 +
              params[7] * t3 +
              params[8] * a1 +
              params[9] * a2)
    
    return result
# EVOLVE-BLOCK-END
