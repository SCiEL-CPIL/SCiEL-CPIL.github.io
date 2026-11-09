"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # New Approach: Small Neural Network (MLP) with 3 hidden neurons.
    # Parameters:
    # 0: w1 (weight to neuron 1)
    # 1: b1 (bias to neuron 1)
    # 2: w2 (weight to neuron 2)
    # 3: b2 (bias to neuron 2)
    # 4: w3 (weight to neuron 3)
    # 5: b3 (bias to neuron 3)
    # 6: v1 (output weight for neuron 1)
    # 7: v2 (output weight for neuron 2)
    # 8: v3 (output weight for neuron 3)
    # 9: c  (output bias)
    
    w1, b1, w2, b2, w3, b3, v1, v2, v3, c = params
    
    # Compute hidden layer activations using tanh
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    h3 = np.tanh(w3 * x + b3)
    
    # Compute output
    return v1 * h1 + v2 * h2 + v3 * h3 + c
# EVOLVE-BLOCK-END
