import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(b)
    
    # Normalize inputs to [-1, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = 2 * (b - 1.0) / (4.998 - 1.0) - 1
    s_norm = 2 * (s - 1.0) / (4.999 - 1.0) - 1
    temp_norm = 2 * (temp - 1.002) / (39.98 - 1.002) - 1
    ph_norm = 2 * (pH - 2.0) / (12.0 - 2.0) - 1
    
    # Clip to [-1, 1] to avoid extrapolation issues during fitting
    b_norm = np.clip(b_norm, -1, 1)
    s_norm = np.clip(s_norm, -1, 1)
    temp_norm = np.clip(temp_norm, -1, 1)
    ph_norm = np.clip(ph_norm, -1, 1)
    
    # Legendre Polynomials
    # L0 = 1
    # L1 = x
    # L2 = 0.5 * (3*x^2 - 1)
    # L3 = 0.5 * (5*x^3 - 3*x)
    
    # Compute L1 and L2 for all variables
    b_L1 = b_norm
    s_L1 = s_norm
    T_L1 = temp_norm
    pH_L1 = ph_norm
    
    b_L2 = 0.5 * (3 * b_norm**2 - 1)
    s_L2 = 0.5 * (3 * s_norm**2 - 1)
    T_L2 = 0.5 * (3 * temp_norm**2 - 1)
    pH_L2 = 0.5 * (3 * ph_norm**2 - 1)
    
    # Compute L3 for temperature (most likely to have complex non-monotonic behavior)
    T_L3 = 0.5 * (5 * temp_norm**3 - 3 * temp_norm)
    
    # Compute L3 for pH
    pH_L3 = 0.5 * (5 * ph_norm**3 - 3 * ph_norm)
    
    # Strategy: Use Legendre polynomials up to 3rd order for key variables
    # and include critical interactions.
    # 
    # Terms:
    # p[0]: constant
    # p[1]: b_L1 (linear b)
    # p[2]: s_L1 (linear s)
    # p[3]: T_L1 (linear T)
    # p[4]: pH_L1 (linear pH)
    # p[5]: T_L2 (quadratic T)
    # p[6]: pH_L2 (quadratic pH)
    # p[7]: T_L3 (cubic T)
    # p[8]: pH_L3 (cubic pH)
    # p[9]: T_L1 * pH_L1 (interaction T-pH)
    
    result = (p[0] 
              + p[1] * b_L1 
              + p[2] * s_L1 
              + p[3] * T_L1 
              + p[4] * pH_L1 
              + p[5] * T_L2 
              + p[6] * pH_L2 
              + p[7] * T_L3 
              + p[8] * pH_L3 
              + p[9] * T_L1 * pH_L1)
              
    return result
