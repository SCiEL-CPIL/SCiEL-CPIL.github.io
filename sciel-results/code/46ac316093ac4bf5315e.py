"""
Strategy: A trigonometric harmonic basis with a linear baseline.

The model uses a linear baseline to capture the mean trend and a set of
sine and cosine terms to capture periodic or wave-like dependencies.
This representation is structurally orthogonal to the previous Gaussian
basis and can model global oscillatory behaviors.

Model:
  f(x, y, z) = c0 + c1*x + c2*y + c3*z
               + w1 * sin(x + y)
               + w2 * cos(y + z)
               + w3 * sin(z + x)
               + w4 * sin(2x)
               + w5 * cos(2y)
               + w6 * sin(2z)

Parameters (10 total):
  0: c0       (intercept)
  1: c1       (x coefficient)
  2: c2       (y coefficient)
  3: c3       (z coefficient)
  4: w1       (weight for sin(x+y))
  5: w2       (weight for cos(y+z))
  6: w3       (weight for sin(z+x))
  7: w4       (weight for sin(2x))
  8: w5       (weight for cos(2y))
  9: w6       (weight for sin(2z))
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a linear baseline plus trigonometric harmonics.
    """
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    w1, w2, w3, w4, w5, w6 = params[4], params[5], params[6], params[7], params[8], params[9]
    
    # Linear baseline
    linear = c0 + c1 * x + c2 * y + c3 * z
    
    # Trigonometric terms
    # Cross terms (interactions)
    term1 = w1 * np.sin(x + y)
    term2 = w2 * np.cos(y + z)
    term3 = w3 * np.sin(z + x)
    
    # Individual higher-frequency terms
    term4 = w4 * np.sin(2.0 * x)
    term5 = w5 * np.cos(2.0 * y)
    term6 = w6 * np.sin(2.0 * z)
    
    return linear + term1 + term2 + term3 + term4 + term5 + term6
