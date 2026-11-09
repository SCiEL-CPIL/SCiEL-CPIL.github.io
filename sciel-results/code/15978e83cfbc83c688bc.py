"""Combined approach: nonlinear parametric model with multiple localized bumps."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model with 10 parameters controlling:
    - Three Gaussian bumps: center, width, amplitude (9 params)
    - Global constant offset (1 param)
    
    This creates a rich nonlinear landscape for BFGS to navigate,
    allowing the model to fit functions with multiple peaks/bumps.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -10, 10)  # safety
    
    # Extract parameters
    # Params: [amp1, c1, w1, amp2, c2, w2, amp3, c3, w3, offset]
    amp1 = params[0]
    c1 = params[1]
    w1 = np.abs(params[2]) + 0.1  # width must be positive
    
    amp2 = params[3]
    c2 = params[4]
    w2 = np.abs(params[5]) + 0.1
    
    amp3 = params[6]
    c3 = params[7]
    w3 = np.abs(params[8]) + 0.1
    
    offset = params[9]
    
    # Three Gaussian bumps
    g1 = amp1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    g2 = amp2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    g3 = amp3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    # Combine
    y = g1 + g2 + g3 + offset
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
