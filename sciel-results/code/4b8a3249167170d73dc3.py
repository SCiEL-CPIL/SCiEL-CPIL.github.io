import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to roughly [0, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    ph_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip to [0, 1] to avoid extrapolation issues during fitting
    b_norm = np.clip(b_norm, 0, 1)
    s_norm = np.clip(s_norm, 0, 1)
    temp_norm = np.clip(temp_norm, 0, 1)
    ph_norm = np.clip(ph_norm, 0, 1)
    
    # Strategy: A constrained 4th-order multivariate polynomial.
    # We cannot fit a full 4th order polynomial (which would have many more terms) with only 10 parameters.
    # Instead, we select key terms that allow for:
    # 1. Constant offset
    # 2. Linear effects for all 4 variables
    # 3. Quadratic effects for temp and pH (to capture bell-shaped/optimal dependencies)
    # 4. Key interaction terms (s*temp, b*s)
    
    # Term mapping:
    # p[0]: constant
    # p[1]: b_norm
    # p[2]: s_norm
    # p[3]: temp_norm
    # p[4]: ph_norm
    # p[5]: temp_norm**2 (quadratic for temp)
    # p[6]: ph_norm**2 (quadratic for pH)
    # p[7]: s_norm * temp_norm (interaction)
    # p[8]: b_norm * s_norm (interaction)
    # p[9]: b_norm**2 (quadratic for b, or could be another interaction)
    
    result = (p[0] 
              + p[1] * b_norm 
              + p[2] * s_norm 
              + p[3] * temp_norm 
              + p[4] * ph_norm 
              + p[5] * temp_norm**2 
              + p[6] * ph_norm**2 
              + p[7] * s_norm * temp_norm 
              + p[8] * b_norm * s_norm 
              + p[9] * b_norm**2)
              
    return result
