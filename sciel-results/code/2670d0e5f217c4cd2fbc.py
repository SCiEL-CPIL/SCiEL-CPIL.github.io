import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=float)
    
    p = params
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = np.clip((b - 1.0) / (4.998 - 1.0), 0, 1)
    s_norm = np.clip((s - 1.0) / (4.999 - 1.0), 0, 1)
    temp_norm = np.clip((temp - 1.002) / (39.98 - 1.002), 0, 1)
    ph_norm = np.clip((pH - 2.0) / (12.0 - 2.0), 0, 1)
    
    # Strategy:
    # Combine the structural idea of "Temperature as a modulator" (from inspirations 2 & 3)
    # with the refinement of using interactions that are biologically plausible.
    #
    # The current program (score 1.1513) used Legendre polynomials up to L2 for all variables
    # and one interaction T*pH.
    #
    # Inspirations 2 & 3 (score 1.1804) used:
    # - Linear terms for all 4 variables
    # - Quadratic terms for Temp and pH (capturing optimal ranges)
    # - Interactions: T*pH, b*T, s*T
    #
    # This structure is superior because:
    # 1. E. coli growth has an optimal temperature range (inverted U), captured by T^2.
    # 2. pH also has an optimal range, captured by pH^2.
    # 3. Temperature modulates the effect of nutrients (b, s), captured by b*T and s*T.
    # 4. Temperature and pH interact in affecting enzyme activity, captured by T*pH.
    #
    # We will adopt this structure directly, as it scored higher in unpaired comparisons.
    # We keep the [0,1] normalization as it is standard for polynomial fitting.
    
    # Terms:
    # p[0]: constant
    # p[1]: b_norm
    # p[2]: s_norm
    # p[3]: temp_norm
    # p[4]: ph_norm
    # p[5]: temp_norm**2
    # p[6]: ph_norm**2
    # p[7]: temp_norm * ph_norm
    # p[8]: b_norm * temp_norm
    # p[9]: s_norm * temp_norm
    
    result = (p[0] 
              + p[1] * b_norm 
              + p[2] * s_norm 
              + p[3] * temp_norm 
              + p[4] * ph_norm 
              + p[5] * temp_norm**2 
              + p[6] * ph_norm**2 
              + p[7] * temp_norm * ph_norm 
              + p[8] * b_norm * temp_norm 
              + p[9] * s_norm * temp_norm)
              
    return result
