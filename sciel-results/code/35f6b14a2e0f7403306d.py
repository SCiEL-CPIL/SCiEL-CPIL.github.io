"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A 3-hidden-unit tanh MLP that can approximate smooth non-linear functions.
    params (length 10):
      p0, p1: weight, bias for hidden unit 0
      p2, p3: weight, bias for hidden unit 1
      p4, p5: weight, bias for hidden unit 2
      p6, p7, p8: output weights for hidden units 0, 1, 2
      p9: output bias
    """
    # Ensure x is a 1-D array
    x = np.asarray(x, dtype=np.float64).ravel()

    # Hidden layer activations
    z0 = np.tanh(params[0] * x + params[1])
    z1 = np.tanh(params[2] * x + params[3])
    z2 = np.tanh(params[4] * x + params[5])

    # Output
    return params[6] * z0 + params[7] * z1 + params[8] * z2 + params[9]
# EVOLVE-BLOCK-END
