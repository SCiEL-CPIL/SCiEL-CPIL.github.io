"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh neural network where params are output-layer weights.
    The model is linear in params, making it suitable for BFGS optimization.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 20 neurons, tanh activation, fixed weights
    - Hidden layer 2: 10 neurons, tanh activation, fixed weights
    - Output: linear combination of h2 features with params[0:10]
    
    This provides universal approximation through fixed internal non-linear
    transformations while remaining linear in the optimizable variables.
    """
    # Fixed random weights for hidden layers (deterministic, well-scaled)
    rng = np.random.RandomState(42)
    
    n_hidden1 = 20
    n_hidden2 = 10
    
    # Hidden layer 1: input (x, y) -> 20 neurons
    # Use Xavier/Glorot initialization scaled for tanh
    W1 = rng.randn(n_hidden1, 2) * np.sqrt(1.0 / 2)
    b1 = rng.randn(n_hidden1) * 0.1
    
    # Hidden layer 2: 20 neurons -> 10 neurons
    W2 = rng.randn(n_hidden2, n_hidden1) * np.sqrt(1.0 / n_hidden1)
    b2 = rng.randn(n_hidden2) * 0.1
    
    # Compute hidden layer activations
    # x and y are 1D arrays of length n
    z1 = x[:, None] * W1[None, :, 0] + y[:, None] * W1[None, :, 1] + b1[None, :]
    h1 = np.tanh(z1)  # shape: (n, 20)
    
    z2 = h1 @ W2.T + b2[None, :]
    h2 = np.tanh(z2)  # shape: (n, 10)
    
    # Output: linear combination of all 10 h2 features with params
    out = h2 @ params
    
    return out
# EVOLVE-BLOCK-END
