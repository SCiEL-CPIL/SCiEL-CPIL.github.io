# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a small MLP (1-2-1) with RBF basis functions.
    
    Parameters (length 10):
    - params[0:2]: Weights from input to hidden layer (w1_0, w1_1)
    - params[2:4]: Biases for hidden layer (b1_0, b1_1)
    - params[4:6]: Weights from hidden to output (w2_0, w2_1)
    - params[6]:   Output bias (b2)
    - params[7:10]: RBF coefficients (3 terms)
    
    The MLP captures global non-linear trends while RBF terms capture local variations.
    Input x is scaled to roughly [-1, 1] for numerical stability.
    RBF centers are fixed at 1.5, 3.0, 4.5 with sigma=1.2.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    x_scaled = (x - 3.05) / 3.0
    
    # --- MLP Part (1-2-1) ---
    # params[0:2]: w1 (weights input to hidden)
    # params[2:4]: b1 (biases hidden)
    # params[4:6]: w2 (weights hidden to output)
    # params[6]:   b2 (output bias)
    
    w1_0, w1_1 = params[0], params[1]
    b1_0, b1_1 = params[2], params[3]
    w2_0, w2_1 = params[4], params[5]
    b2 = params[6]
    
    # Hidden layer activations (tanh)
    h0 = np.tanh(w1_0 * x_scaled + b1_0)
    h1 = np.tanh(w1_1 * x_scaled + b1_1)
    
    # Output layer
    mlp_out = w2_0 * h0 + w2_1 * h1 + b2
    
    # --- RBF Part (3 terms) ---
    # params[7:10]: RBF coefficients
    c1, c2, c3 = params[7], params[8], params[9]
    
    # Fixed RBF centers and sigma
    centers = np.array([1.5, 3.0, 4.5])
    sigma = 1.2
    
    # Compute RBF values: exp(-(x - c_j)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (n, 3)
    
    # RBF contribution
    rbf_out = rbf_values @ np.array([c1, c2, c3])
    
    # Combine MLP and RBF
    y = mlp_out + rbf_out
    
    return y
# EVOLVE-BLOCK-END
