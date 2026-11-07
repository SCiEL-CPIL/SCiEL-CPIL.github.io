"""
Strategy: A flexible neural-network-like model with tanh activations.

The model uses a single hidden layer with tanh non-linearity, allowing it to
capture complex non-linear interactions between the three inputs. The architecture
is:
  f(x, y, z) = b0 + sum_i (b_i * tanh(a_i0 + a_i1*x + a_i2*y + a_i3*z))

With 10 parameters, we allocate:
  0: b0       (output bias)
  1: b1       (weight for hidden unit 1)
  2: b2       (weight for hidden unit 2)
  3: a10      (bias for hidden unit 1)
  4: a11      (x weight for hidden unit 1)
  5: a12      (y weight for hidden unit 1)
  6: a13      (z weight for hidden unit 1)
  7: a20      (bias for hidden unit 2)
  8: a21      (x weight for hidden unit 2)
  9: a22      (y weight for hidden unit 2)

We fix a23 (z weight for hidden unit 2) to a13 to save one parameter, or we can
use a different allocation. Let me reconsider to use all parameters effectively.

Better allocation with 2 hidden units:
  0: b0       (output bias)
  1: b1       (weight for hidden unit 1)
  2: b2       (weight for hidden unit 2)
  3: a10      (bias for hidden unit 1)
  4: a11      (x weight for hidden unit 1)
  5: a12      (y weight for hidden unit 1)
  6: a13      (z weight for hidden unit 1)
  7: a20      (bias for hidden unit 2)
  8: a21      (x weight for hidden unit 2)
  9: a22      (y weight for hidden unit 2)

We fix a23 = a13 (share z weight between units) to keep 10 params.
Alternatively, we can use a linear baseline plus 2 tanh units with fewer params each.

Let's use:
  0: c0       (linear intercept)
  1: c1       (linear x)
  2: c2       (linear y)
  3: c3       (linear z)
  4: w1       (tanh unit 1 weight)
  5: a10      (tanh unit 1 bias)
  6: a11      (tanh unit 1 x)
  7: a12      (tanh unit 1 y)
  8: w2       (tanh unit 2 weight)
  9: a20      (tanh unit 2 bias)

And fix a13=1.0, a21=1.0, a22=1.0 for the second unit, or make it symmetric.

Actually, a cleaner approach: use a sum of two tanh functions with shared structure.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a linear baseline plus two tanh-based hidden units.
    
    Parameters:
      0: c0  - intercept
      1: c1  - x linear coeff
      2: c2  - y linear coeff
      3: c3  - z linear coeff
      4: w1  - weight for tanh unit 1
      5: a10 - bias for tanh unit 1
      6: a11 - x weight for tanh unit 1
      7: a12 - y weight for tanh unit 1
      8: w2  - weight for tanh unit 2
      9: a20 - bias for tanh unit 2
    """
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    w1, a10, a11, a12 = params[4], params[5], params[6], params[7]
    w2, a20 = params[8], params[9]
    
    # Linear baseline
    linear = c0 + c1 * x + c2 * y + c3 * z
    
    # Tanh unit 1: tanh(a10 + a11*x + a12*y + 1.0*z)
    h1 = np.tanh(a10 + a11 * x + a12 * y + 1.0 * z)
    
    # Tanh unit 2: tanh(a20 + 1.0*x + 1.0*y + 1.0*z)
    h2 = np.tanh(a20 + 1.0 * x + 1.0 * y + 1.0 * z)
    
    return linear + w1 * h1 + w2 * h2
