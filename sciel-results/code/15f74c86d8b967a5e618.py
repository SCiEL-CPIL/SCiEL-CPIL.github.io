import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Feedforward neural network-style model with fixed random projection weights
    and 10 learnable parameters (5 hidden gains + 5 output weights).
    
    Parameters (10 total):
    p0-p4: Hidden layer activation gains (5 neurons)
    p5-p9: Output layer weights (5 neurons)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed random weights for hidden layer (4 input features -> 5 hidden neurons)
    # These are fixed, non-learnable random projections
    # Using a deterministic pseudo-random matrix based on fixed seeds
    W_hidden = np.array([
        [0.5, -0.3, 0.7, 0.1],
        [-0.2, 0.8, -0.4, 0.6],
        [0.9, 0.1, -0.5, 0.3],
        [-0.6, 0.4, 0.2, -0.7],
        [0.3, -0.9, 0.5, -0.4]
    ])
    
    # Input features: [b, s, temp, pH]
    X = np.column_stack([b, s, temp, pH])
    
    # Hidden layer: fixed linear combination + learnable gain * tanh
    H = X @ W_hidden.T  # shape: (n_samples, 5)
    
    # Learnable gains scale the hidden activations
    gains = np.array([p0, p1, p2, p3, p4])
    H_activated = np.tanh(gains * H)  # shape: (n_samples, 5)
    
    # Output layer: weighted sum of hidden activations + bias (implicit in weights)
    # Use learnable output weights
    w_out = np.array([p5, p6, p7, p8, p9])
    
    # Final prediction: weighted sum of hidden layer outputs
    growth = H_activated @ w_out
    
    return growth
