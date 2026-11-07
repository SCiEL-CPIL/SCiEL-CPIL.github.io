"""Symbolic regression with 2 inputs: fixed-architecture neural network with tanh activations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-architecture neural network with tanh activations.
    
    Architecture:
    - Input: x, y (2 features)
    - Hidden Layer 1: 8 neurons, tanh activation
    - Hidden Layer 2: 8 neurons, tanh activation
    - Output: 1 neuron, linear activation
    
    The 10 parameters are used as follows:
    - params[0:8]: weights from Hidden Layer 2 to output (8 weights)
    - params[8]: bias for output layer
    - params[9]: scale factor for output
    
    Fixed internal weights are deterministic (seeded), providing non-linear feature extraction.
    The model is linear in params, ensuring BFGS converges efficiently.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Fixed internal weights (deterministic)
    rng = np.random.RandomState(42)
    
    # Hidden Layer 1: 8 neurons
    W1 = rng.randn(2, 8) * 0.5  # (2, 8)
    b1 = rng.randn(8) * 0.1     # (8,)
    
    # Hidden Layer 2: 8 neurons
    W2 = rng.randn(8, 8) * 0.5  # (8, 8)
    b2 = rng.randn(8) * 0.1     # (8,)
    
    # Forward pass
    # Hidden Layer 1
    h1 = np.tanh(x[:, 0:1] @ W1[0:1, :] + x[:, 1:2] @ W1[1:2, :] + b1)
    
    # Hidden Layer 2
    h2 = np.tanh(h1 @ W2 + b2)
    
    # Output layer: linear combination of h2 with params[0:8] as weights
    # params[8] is bias, params[9] is scale
    output = h2 @ params[0:8] + params[8]
    output = output * params[9]
    
    return output
# EVOLVE-BLOCK-END
