"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model with 10 parameters.
    
    Parameters are used to define feature locations, scales, and amplitudes:
    - p[0], p[1]: center and amplitude of Gaussian 1
    - p[2], p[3]: width and amplitude of Gaussian 2
    - p[4], p[5]: frequency and amplitude of sin component
    - p[6], p[7]: frequency and amplitude of cos component
    - p[8], p[9]: linear slope and constant offset
    
    This non-linear model lets BFGS place features adaptively.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Clamp parameters to safe ranges to avoid numerical issues
    # Centers in [-3, 3], widths in [0.1, 3], amplitudes in [-10, 10]
    c1 = np.clip(p[0], -3.0, 3.0)
    a1 = np.clip(p[1], -10.0, 10.0)
    w1 = np.clip(p[2], 0.1, 3.0)
    a2 = np.clip(p[3], -10.0, 10.0)
    f1 = np.clip(p[4], 0.1, 5.0)
    asin = np.clip(p[5], -10.0, 10.0)
    f2 = np.clip(p[6], 0.1, 5.0)
    acos = np.clip(p[7], -10.0, 10.0)
    slope = np.clip(p[8], -10.0, 10.0)
    const = np.clip(p[9], -10.0, 10.0)
    
    # Gaussian 1: a1 * exp(-(x - c1)^2 / (2*w1^2))
    g1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    
    # Gaussian 2: a2 * exp(-x^2 / (2*w1^2))  (centered at 0, using w1 for width)
    g2 = a2 * np.exp(-(x ** 2) / (2.0 * w1 ** 2))
    
    # Sin component: asin * sin(f1 * x)
    s1 = asin * np.sin(f1 * x)
    
    # Cos component: acos * cos(f2 * x)
    c1_trig = acos * np.cos(f2 * x)
    
    # Linear part
    linear = slope * x + const
    
    result = g1 + g2 + s1 + c1_trig + linear
    
    return result
# EVOLVE-BLOCK-END
