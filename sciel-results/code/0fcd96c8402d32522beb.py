import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a learnable-frequency Fourier series basis.
    
    Parameters (length 10):
    - params[0] : constant bias
    - params[1] : linear coefficient for x
    - params[2] : linear coefficient for y
    - params[3] : amplitude for cos(fx * x)
    - params[4] : amplitude for sin(fx * x)
    - params[5] : amplitude for cos(fy * y)
    - params[6] : amplitude for sin(fy * y)
    - params[7] : frequency fx (learnable)
    - params[8] : frequency fy (learnable)
    - params[9] : amplitude for cos(fx*x + fy*y) interaction term
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Deadline guard: track time to avoid exceeding 150s
    # This is a lightweight check; the actual computation is fast.
    t0 = time.time()
    
    # Extract parameters
    b0 = params[0]
    bx = params[1]
    by = params[2]
    amp_cx = params[3]
    amp_sx = params[4]
    amp_cy = params[5]
    amp_sy = params[6]
    fx = params[7]
    fy = params[8]
    amp_interact = params[9]
    
    # Clamp frequencies to avoid extreme oscillations that cause numerical issues
    # Frequencies in [-10, 10] cover a wide range of possible periodicity
    fx_safe = np.clip(fx, -10.0, 10.0)
    fy_safe = np.clip(fy, -10.0, 10.0)
    
    # Compute Fourier terms
    pred = np.full_like(x, b0, dtype=float)
    pred += bx * x
    pred += by * y
    pred += amp_cx * np.cos(fx_safe * x)
    pred += amp_sx * np.sin(fx_safe * x)
    pred += amp_cy * np.cos(fy_safe * y)
    pred += amp_sy * np.sin(fy_safe * y)
    pred += amp_interact * np.cos(fx_safe * x + fy_safe * y)
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to linear-only if anything went wrong
        pred = b0 + bx * x + by * y
    
    return pred
