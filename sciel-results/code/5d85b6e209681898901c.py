"""Symbolic regression with 3 inputs – single-hidden-layer tanh network basis."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a single-hidden-layer tanh network with 10 parameters.
    
    Architecture:
      - Input: (x, y, z)
      - Hidden layer: 3 units, tanh activation
        h_i = tanh(w_i1*x + w_i2*y + w_i3*z + b_i)  for i in {0,1,2}
        But we only have 10 params total, so we need to be careful.
      
    Let's allocate 10 params:
      - 3 weights for x in hidden layer: p0, p1, p2
      - 3 weights for y in hidden layer: p3, p4, p5
      - 3 weights for z in hidden layer: p6, p7, p8
      - 1 bias shared or output bias: p9
      
    Actually, let's think differently. A simple approach:
      - 3 hidden units, each with 3 input weights = 9 weights
      - 1 output bias = 1 param
      Total = 10 params.
      
    But that leaves no output weights for the hidden units. Let's use a different split:
      - 2 hidden units: 2*3 = 6 input weights
      - 2 output weights for the 2 hidden units
      - 2 biases for the 2 hidden units
      - 1 output bias
      Total = 6 + 2 + 2 + 1 = 11... too many.
      
    Let's try:
      - 3 hidden units, but share some structure.
      
    Simpler approach: Use 10 params as follows:
      h0 = tanh(p0*x + p1*y + p2*z)
      h1 = tanh(p3*x + p4*y + p5*z)
      h2 = tanh(p6*x + p7*y + p8*z)
      output = p9 * (h0 + h1 + h2)
      
    This gives us 9 weights for the hidden layer and 1 scaling factor for the output.
    This is a valid 10-parameter model.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute hidden layer activations
    h0 = np.tanh(p0 * x + p1 * y + p2 * z)
    h1 = np.tanh(p3 * x + p4 * y + p5 * z)
    h2 = np.tanh(p6 * x + p7 * y + p8 * z)
    
    # Output: linear combination of hidden units with a single scale
    return p9 * (h0 + h1 + h2)
