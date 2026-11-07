import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # 4th-order multivariate polynomial with 10 coefficients
    # Terms: 1, b, s, t, ph, b*s, b*t, b*ph, s*t, s*ph
    # This gives us 10 terms total, using all 10 parameters
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm 
         + p[5] * b_norm * s_norm 
         + p[6] * b_norm * t_norm 
         + p[7] * b_norm * ph_norm 
         + p[8] * s_norm * t_norm 
         + p[9] * s_norm * ph_norm)
    
    return y
