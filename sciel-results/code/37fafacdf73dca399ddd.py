import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: start timer
    t0 = time.time()
    DEADLINE = 140.0  # 10s margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Compute basis functions
    # Combine tanh-based (bounded, smooth) with polynomial and trigonometric
    # for broader coverage of possible hidden functions
    
    # Polynomial basis (low order for stability)
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * y
    b5 = x * x
    b6 = y * y
    
    # Tanh-based bounded non-linearities
    t1 = np.tanh(2.0 * x)
    t2 = np.tanh(2.0 * y)
    t3 = np.tanh(2.0 * x * y)
    
    # Trigonometric for periodic structure
    s1 = np.sin(x)
    c1 = np.cos(y)
    
    # Build prediction as linear combination
    # Use all 10 params
    pred = (p[0] * b1 + 
            p[1] * b2 + 
            p[2] * b3 + 
            p[3] * b4 + 
            p[4] * b5 + 
            p[5] * b6 + 
            p[6] * t1 + 
            p[7] * t2 + 
            p[8] * t3 + 
            p[9] * s1 * c1)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        return pred
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Replace non-finite with large finite value
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
