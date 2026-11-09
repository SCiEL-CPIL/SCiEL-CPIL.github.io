import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed trigonometric-polynomial basis for nonlinear oscillator.
    
    Structural change: replaces pure polynomial with a hybrid basis that
    includes trigonometric terms (sin, cos) to capture oscillatory structure,
    plus polynomial terms for non-periodic components.
    
    Basis terms (10 total):
    1. 1 (bias)
    2. x
    3. v
    4. x^2
    5. x*v
    6. sin(x)
    7. cos(x) - 1
    8. sin(v)
    9. cos(v) - 1
    10. x*sin(x)
    
    Normalization applied to inputs for numerical conditioning.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    # Normalization constants based on training ranges
    x_mean = -0.02265
    x_scale = 0.55015
    v_mean = 0.05335
    v_scale = 0.22875
    
    # Normalize inputs
    x_n = (x - x_mean) / x_scale
    v_n = (v - v_mean) / v_scale
    
    p = params
    
    # Construct mixed basis terms
    # Polynomial terms
    t1 = np.ones_like(x_n)
    t2 = x_n
    t3 = v_n
    t4 = x_n * x_n
    t5 = x_n * v_n
    
    # Trigonometric terms (capture oscillatory structure)
    t6 = np.sin(x_n)
    t7 = np.cos(x_n) - 1.0
    t8 = np.sin(v_n)
    t9 = np.cos(v_n) - 1.0
    
    # Mixed term: position-modulated oscillation
    t10 = x_n * np.sin(x_n)
    
    # Combine all terms
    a = (p[0] * t1 + 
         p[1] * t2 + 
         p[2] * t3 + 
         p[3] * t4 + 
         p[4] * t5 +
         p[5] * t6 + 
         p[6] * t7 + 
         p[7] * t8 + 
         p[8] * t9 + 
         p[9] * t10)
    
    # Check deadline - if approaching limit, return current result immediately
    if time.time() - t_start > DEADLINE:
        return a
    
    return a
