"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural network with two hidden layers using tanh activations.
    The 10 parameters are the weights of the final output layer (and bias),
    making the model linear in the optimizable variables for BFGS.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 5 neurons, tanh activation, fixed weights
    - Hidden layer 2: 5 neurons, tanh activation, fixed weights
    - Output layer: linear combination of hidden layer 2 outputs with params[0:5] as weights
    - params[5:10] used as additional linear terms: params[5]*x + params[6]*y + params[7]
      plus interactions to enhance expressiveness while keeping linearity in params.
    
    Actually, to keep it clean and ensure BFGS works well, let's design:
    - Hidden layer 1: 5 neurons with fixed random weights (seeded for reproducibility)
    - Hidden layer 2: 5 neurons with fixed random weights
    - Output: params[0]*h2[0] + params[1]*h2[1] + ... + params[4]*h2[4] + params[5]*x + params[6]*y + params[7] + params[8]*x*y + params[9]
    
    This keeps the model linear in all 10 parameters.
    """
    # Fixed random weights for hidden layers (deterministic)
    rng = np.random.RandomState(42)
    
    # Hidden layer 1: input (x, y) -> 5 neurons
    W1 = rng.randn(5, 2) * np.sqrt(2.0 / 2)  # He-like init
    b1 = np.zeros(5)
    
    # Hidden layer 2: 5 neurons -> 5 neurons
    W2 = rng.randn(5, 5) * np.sqrt(2.0 / 5)
    b2 = np.zeros(5)
    
    # Compute hidden layer activations
    z1 = x[:, None] * W1[None, :, 0] + y[:, None] * W1[None, :, 1] + b1[None, :]
    h1 = np.tanh(z1)  # shape: (n, 5)
    
    z2 = h1 @ W2.T + b2[None, :]
    h2 = np.tanh(z2)  # shape: (n, 5)
    
    # Output: linear combination of h2 and input features
    # params[0:5] weights for h2, params[5] for x, params[6] for y, params[7] for x*y, params[8] for x^2, params[9] bias
    out = (h2 * params[0:5][None, :]).sum(axis=1)
    out += params[5] * x
    out += params[6] * y
    out += params[7] * x * y
    out += params[8] * x**2
    out += params[9]
    
    return out
# EVOLVE-BLOCK-END
