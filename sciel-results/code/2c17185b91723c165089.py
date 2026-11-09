"""Symbolic regression with 1 input - Small Tanh Neural Network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single hidden layer tanh network with 3 neurons.
    
    y = c + sum_{j=1}^{3} w_j * tanh(a_j * x + b_j)
    
    10 parameters:
      params[0]: c (bias)
      params[1:4]: w_1, w_2, w_3 (output weights)
      params[4:7]: a_1, a_2, a_3 (input scales/slopes)
      params[7:10]: b_1, b_2, b_3 (hidden biases/shifts)
    
    This model allows the optimizer to adjust the shape and position of the
    non-linear features, offering more flexibility than a fixed basis.
    """
    c = params[0]
    w1, w2, w3 = params[1], params[2], params[3]
    a1, a2, a3 = params[4], params[5], params[6]
    b1, b2, b3 = params[7], params[8], params[9]
    
    # Compute hidden layer activations
    # Using np.tanh which is stable and smooth
    h1 = np.tanh(a1 * x + b1)
    h2 = np.tanh(a2 * x + b2)
    h3 = np.tanh(a3 * x + b3)
    
    # Combine
    y = c + w1 * h1 + w2 * h2 + w3 * h3
    
    return y
# EVOLVE-BLOCK-END
