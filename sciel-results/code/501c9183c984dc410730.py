"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using a small neural network with fixed architecture.
    
    We use 10 parameters to define a 3-hidden-unit ReLU network:
    - 3 input weights (params[0:3])
    - 3 hidden biases (params[3:6])
    - 3 output weights (params[6:9])
    - 1 output bias (params[9])
    
    Total: 10 parameters.
    """
    # Extract parameters
    w1_0, w1_1, w1_2 = params[0], params[1], params[2]
    b1_0, b1_1, b1_2 = params[3], params[4], params[5]
    w2_0, w2_1, w2_2 = params[6], params[7], params[8]
    b2 = params[9]
    
    # Hidden layer computations
    z1 = w1_0 * x + b1_0
    z2 = w1_1 * x + b1_1
    z3 = w1_2 * x + b1_2
    
    # ReLU activation
    a1 = np.maximum(0, z1)
    a2 = np.maximum(0, z2)
    a3 = np.maximum(0, z3)
    
    # Output layer
    y = w2_0 * a1 + w2_1 * a2 + w2_2 * a3 + b2
    
    return y
# EVOLVE-BLOCK-END
