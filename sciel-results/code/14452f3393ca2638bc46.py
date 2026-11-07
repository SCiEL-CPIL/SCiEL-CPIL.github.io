# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Composite basis: polynomial + trigonometric + tanh saturation.
    10 parameters total:
      params[0..3]: polynomial coefficients (1, x, x^2, x^3)
      params[4..7]: trigonometric coefficients (sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x))
      params[8..9]: tanh saturation (tanh(x), tanh(2*x))
    
    This is a linear model in params, so BFGS (or any convex optimizer) converges globally.
    The basis covers:
      - Polynomial trends (up to cubic)
      - Periodic features (frequency 1 and 2, i.e. period 2 and 1)
      - Saturation / sigmoid-like features
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Polynomial basis: 1, x, x^2, x^3
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    # Trig basis: sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x)
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    
    # Tanh saturation: tanh(x), tanh(2*x)
    p8 = params[8]
    p9 = params[9]
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    
    pi_x = np.pi * x
    sin_pi_x = np.sin(pi_x)
    cos_pi_x = np.cos(pi_x)
    sin_2pi_x = np.sin(2.0 * pi_x)
    cos_2pi_x = np.cos(2.0 * pi_x)
    
    tanh_x = np.tanh(x)
    tanh_2x = np.tanh(2.0 * x)
    
    # Linear combination
    result = (p0 * np.ones_like(x) +
              p1 * x +
              p2 * x2 +
              p3 * x3 +
              p4 * sin_pi_x +
              p5 * cos_pi_x +
              p6 * sin_2pi_x +
              p7 * cos_2pi_x +
              p8 * tanh_x +
              p9 * tanh_2x)
    
    return result
# EVOLVE-BLOCK-END
