import numpy as np

# Fixed seed for reproducible internal weights
np.random.seed(42)

# Define internal weights and biases for the neural network
# Input size: 2 (x, y)
# Hidden Layer 1 size: 8
# Hidden Layer 2 size: 10
# Output size: 1 (linear combination of H2 with params)

# W1: (2, 8), b1: (8,)
W1 = np.random.randn(2, 8) * 1.0
b1 = np.random.randn(8) * 0.1

# W2: (8, 10), b2: (10,)
W2 = np.random.randn(8, 10) * 1.0
b2 = np.random.randn(10) * 0.1

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural network model with fixed internal weights and linear output layer.
    The 10 parameters are the weights of the final output layer.
    This makes the model linear in params, suitable for BFGS.
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Stack inputs
    X = np.column_stack((x, y)) # Shape (N, 2)
    
    # Hidden Layer 1
    Z1 = X @ W1 + b1 # Shape (N, 8)
    H1 = np.tanh(Z1)
    
    # Hidden Layer 2
    Z2 = H1 @ W2 + b2 # Shape (N, 10)
    H2 = np.tanh(Z2)
    
    # Output Layer (linear in params)
    # H2 is (N, 10), params is (10,)
    # We want a vector of predictions of shape (N,)
    pred = H2 @ p
    
    return pred
