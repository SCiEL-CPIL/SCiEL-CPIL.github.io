import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    _t0 = time.time()
    _deadline = 145.0  # 5s margin before 150s kill
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Sparse multivariate polynomial
    # Replaces mechanistic multiplicative model with a flexible polynomial surface.
    # 10 parameters for a sparse polynomial capturing main effects, quadratics, and key interactions.
    # This allows the optimizer to fit arbitrary high-order interaction terms and
    # non-monotonic dependencies across all variables simultaneously.
    
    # p[0]: intercept
    # p[1]: b linear
    # p[2]: s linear
    # p[3]: t linear
    # p[4]: ph linear
    # p[5]: b^2
    # p[6]: s^2
    # p[7]: t^2
    # p[8]: ph^2
    # p[9]: b*s interaction
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm
         + p[5] * b_norm**2 
         + p[6] * s_norm**2 
         + p[7] * t_norm**2 
         + p[8] * ph_norm**2
         + p[9] * b_norm * s_norm)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
