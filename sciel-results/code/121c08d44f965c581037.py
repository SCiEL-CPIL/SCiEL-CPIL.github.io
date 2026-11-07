"""Rewritten program with a richer basis function set for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit

    # Construct basis features
    # 10 features:
    # 1: constant
    # 2: x
    # 3: y
    # 4: x^2
    # 5: y^2
    # 6: xy
    # 7: sin(x)
    # 8: sin(y)
    # 9: cos(x)
    # 10: cos(y)
    
    # Ensure x and y are 1D arrays for consistent shape handling
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Calculate features
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x**2
    f5 = y**2
    f6 = x * y
    f7 = np.sin(x)
    f8 = np.sin(y)
    f9 = np.cos(x)
    f10 = np.cos(y)
    
    # Combine with parameters
    # params is an array of length 10
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    result = (p0 * f1 + p1 * f2 + p2 * f3 + p3 * f4 + p4 * f5 + 
              p5 * f6 + p6 * f7 + p7 * f8 + p8 * f9 + p9 * f10)
              
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return a safe constant if numerical issues arise
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
