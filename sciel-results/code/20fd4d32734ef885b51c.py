import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Scaling to [-1, 1]
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Basis functions
    b0 = np.ones_like(x)
    b1 = u
    b2 = w
    b3 = u * u
    b4 = w * w
    b5 = np.sin(u)
    b6 = np.sin(w)
    b7 = np.tanh(u)
    b8 = np.tanh(w)
    b9 = u * w

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
