"""Symbolic regression with 1 input(s) - Rich basis with good conditioning."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions designed to span a wide
    range of smooth functions on [-2, 2].
    
    Basis functions (designed for expressiveness with 10 params):
      0: 1                          (constant)
      1: x                          (linear)
      2: x^2                        (quadratic)
      3: x^3                        (cubic)
      4: x^4                        (quartic)
      5: tanh(x)                    (gentle saturation)
      6: tanh(3*x)                  (steeper saturation)
      7: exp(-x^2)                  (Gaussian bump at center)
      8: sin(x)                     (low-frequency oscillation)
      9: sin(2*x)                   (second harmonic)
    
    This basis combines polynomial, saturating, exponential, and oscillatory
    components to handle diverse target functions.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Clip x to avoid numerical issues in exp for extreme values
    x_c = np.clip(x, -50, 50)
    
    # Compute basis functions
    b0 = np.ones_like(x_c)
    b1 = x_c
    b2 = x_c ** 2
    b3 = x_c ** 3
    b4 = x_c ** 4
    
    # Saturating terms
    b5 = np.tanh(x_c)
    b6 = np.tanh(3.0 * x_c)
    
    # Gaussian bump - use clipped exponent to avoid underflow issues
    exp_arg = -x_c ** 2
    exp_arg = np.maximum(exp_arg, -100)
    b7 = np.exp(exp_arg)
    
    # Sinusoidal terms
    b8 = np.sin(x_c)
    b9 = np.sin(2.0 * x_c)
    
    # Linear combination
    p = params
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
         p[5] * b5 + p[6] * b6 + p[7] * b7 +
         p[8] * b8 + p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
