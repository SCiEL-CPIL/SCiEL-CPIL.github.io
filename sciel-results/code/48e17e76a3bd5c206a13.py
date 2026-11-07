"""Symbolic regression with 1 input; Small Neural Network (1-3-1 MLP)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a small Multilayer Perceptron (MLP).
    
    Architecture: 1 input -> 3 hidden neurons (Tanh) -> 1 output.
    Parameters (length 10):
    - params[0:3]: Weights from input to hidden layer (w1)
    - params[3:6]: Biases for hidden layer (b1)
    - params[6:9]: Weights from hidden to output (w2)
    - params[9]:   Bias for output (b2)
    
    The input x is centered and scaled to [-1, 1] approximately to improve
    optimization stability for BFGS starting from all ones.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure params is a flat array of floats
    params = np.asarray(params, dtype=float).flatten()
    
    # Unpack parameters
    # 3 hidden units
    # Input to Hidden weights (3)
    w1 = params[0:3]
    # Hidden biases (3)
    b1 = params[3:6]
    # Hidden to Output weights (3)
    w2 = params[6:9]
    # Output bias (1)
    b2 = params[9]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    # Center at 3.05, scale by half-range ~ 3.0
    x_scaled = (x - 3.05) / 3.0
    
    # Compute hidden layer activations
    # z1 = x_scaled * w1[0] + b1[0]
    # z2 = x_scaled * w1[1] + b1[1]
    # z3 = x_scaled * w1[2] + b1[2]
    # h = tanh(z)
    
    # Vectorized computation
    # x_scaled shape: (N,)
    # w1 shape: (3,)
    # We need (N, 3) matrix for dot product with w1
    # Alternatively, compute each neuron separately for clarity and speed
    
    h1 = np.tanh(x_scaled * w1[0] + b1[0])
    h2 = np.tanh(x_scaled * w1[1] + b1[1])
    h3 = np.tanh(x_scaled * w1[2] + b1[2])
    
    # Combine hidden units
    y = w2[0] * h1 + w2[1] * h2 + w2[2] * h3 + b2
    
    return y
# EVOLVE-BLOCK-END
