"""Symbolic regression with 1 input; EVOLVE-BLOCK contains the equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural network with a single hidden layer of 10 ReLU units.
    params: array of length 10.
    - params[0]: bias for the output layer
    - params[1:11] would be needed but we only have 10 params.
    
    Actually, let's design with exactly 10 parameters.
    We need: weights from input to hidden, biases for hidden, weights from hidden to output, bias for output.
    
    Let's use 4 hidden units to keep parameter count manageable?
    No, the directive says "single hidden layer of 10 ReLU units". But we only have 10 params total.
    That's not enough for 10 hidden units with input weights and biases.
    
    Let me re-read: "params, an array of length 10".
    So we have exactly 10 parameters to fit.
    
    Let's design a smaller network. 
    Maybe 2 hidden units?
    Input: 1
    Hidden: 2 ReLU units
    Output: 1
    
    Parameters needed:
    - w1, b1 (input to hidden unit 1)
    - w2, b2 (input to hidden unit 2)
    - v1, v2 (hidden to output weights)
    - c (output bias)
    Total: 7 parameters. We have 10.
    
    We can use 3 hidden units:
    - w1, b1, w2, b2, w3, b3 (input to hidden)
    - v1, v2, v3 (hidden to output)
    - c (output bias)
    Total: 10 parameters. Perfect.
    
    Let's structure params as:
    params[0] = w1
    params[1] = b1
    params[2] = w2
    params[3] = b2
    params[4] = w3
    params[5] = b3
    params[6] = v1
    params[7] = v2
    params[8] = v3
    params[9] = c
    
    The function is:
    h1 = ReLU(w1 * x + b1)
    h2 = ReLU(w2 * x + b2)
    h3 = ReLU(w3 * x + b3)
    y = v1 * h1 + v2 * h2 + v3 * h3 + c
    """
    # Extract parameters
    w1, b1, w2, b2, w3, b3, v1, v2, v3, c = params
    
    # Compute hidden layer activations
    h1 = np.maximum(0.0, w1 * x + b1)
    h2 = np.maximum(0.0, w2 * x + b2)
    h3 = np.maximum(0.0, w3 * x + b3)
    
    # Compute output
    y = v1 * h1 + v2 * h2 + v3 * h3 + c
    
    return y
# EVOLVE-BLOCK-END
