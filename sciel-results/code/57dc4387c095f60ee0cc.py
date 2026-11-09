import numpy as np
import time

# Module level time tracking for safety, though the evaluator controls the loop
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    
    # Hybrid model: Linear trend + 4 Gaussian bumps
    # Parameters:
    # p[0]: bias
    # p[1]: slope
    # p[2]: amp1, p[3]: width1_raw
    # p[4]: amp2, p[5]: width2_raw
    # p[6]: amp3, p[7]: width3_raw
    # p[8]: amp4, p[9]: width4_raw
    
    y = params[0] + params[1] * x
    
    # Fixed centers covering the domain [0.1, 6]
    centers = np.array([0.8, 2.3, 3.8, 5.3])
    
    for i in range(4):
        amp = params[2 * i + 2]
        width_raw = params[2 * i + 3]
        
        # Stable positive width calculation
        # Clip to prevent overflow in exp
        w = np.exp(np.clip(width_raw, -5.0, 5.0))
        sigma = 0.1 + w
        
        dist_sq = (x - centers[i]) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma))
        
        y += amp * gaussian
        
    return y
