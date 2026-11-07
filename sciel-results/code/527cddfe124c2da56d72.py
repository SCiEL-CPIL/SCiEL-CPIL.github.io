"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for better conditioning
    t = x / 2.0
    
    # Chebyshev polynomials of the first kind (computed via recurrence for stability)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    # Trigonometric basis with frequencies adapted to the domain [-2, 2]
    # Using pi*x/2 gives period 4, which matches the domain width
    theta1 = (np.pi / 2.0) * x
    theta2 = np.pi * x
    
    sin_t1 = np.sin(theta1)
    cos_t1 = np.cos(theta1)
    sin_t2 = np.sin(theta2)
    cos_t2 = np.cos(theta2)
    
    # Smooth bounded non-linear feature
    tanh_x = np.tanh(x)
    
    # Combine features:
    # 0: bias (T0)
    # 1: T1 (linear)
    # 2: T2 (quadratic)
    # 3: T3 (cubic)
    # 4: T4 (quartic)
    # 5: sin(pi*x/2)
    # 6: cos(pi*x/2)
    # 7: sin(pi*x)
    # 8: cos(pi*x)
    # 9: tanh(x)
    
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * sin_t1 +
              params[6] * cos_t1 +
              params[7] * sin_t2 +
              params[8] * cos_t2 +
              params[9] * tanh_x)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return a large constant to penalize this step
        return np.full_like(x, 1e10)
        
    return result
# EVOLVE-BLOCK-END
