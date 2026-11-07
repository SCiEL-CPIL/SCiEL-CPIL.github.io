import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Scaling
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = (x - (x_min + x_max)/2.0) / (x_range/2.0)
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = (v - (v_min + v_max)/2.0) / (v_range/2.0)

    # Basis
    b0 = np.ones_like(x)
    b1 = u
    b2 = w
    b3 = u * u
    b4 = u * w
    b5 = w * w
    b6 = u * u * u
    b7 = u * u * w
    b8 = w * w * w
    b9 = u * u * u * u

    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
