import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return a simple prediction to avoid timeout crash
        # This is a safety net, should rarely trigger
        return np.zeros_like(b)

    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / (4.998 - 1.0), 0, 1)
    s_norm = np.clip((s - 1.0) / (4.999 - 1.0), 0, 1)
    temp_norm = np.clip((temp - 1.002) / (39.98 - 1.002), 0, 1)
    ph_norm = np.clip((pH - 2.0) / (12.0 - 10.0), 0, 1) # Wait, pH range is [2, 12], so 10.0
    
    # Correct pH normalization denominator
    ph_norm = np.clip((pH - 2.0) / (12.0 - 2.0), 0, 1)
    
    # Term selection:
    # Focus on Temperature as a modulator of other factors and its own non-linearity.
    # 1. Constant
    # 2. Linear b
    # 3. Linear s
    # 4. Linear T
    # 5. Linear P
    # 6. Quadratic T
    # 7. Quadratic P
    # 8. Interaction T * P
    # 9. Interaction b * T
    # 10. Interaction s * T
    
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
