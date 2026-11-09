"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a small neural network structure (3 hidden tanh units + bias)
    to approximate the target function.
    
    Parameters (length 10):
    - params[0:3]  : w1, b1, c1 for unit 1
    - params[3:6]  : w2, b2, c2 for unit 2
    - params[6:9]  : w3, b3, c3 for unit 3
    - params[9]    : bias term
    
    Model: y = bias + c1*tanh(w1*x + b1) + c2*tanh(w2*x + b2) + c3*tanh(w3*x + b3)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    params = np.asarray(params, dtype=float)
    
    # Ensure params has length 10
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(params.shape[0], 10)
        p[:n] = params[:n]
        params = p
        
    # Extract parameters
    w1, b1, c1 = params[0], params[1], params[2]
    w2, b2, c2 = params[3], params[4], params[5]
    w3, b3, c3 = params[6], params[7], params[8]
    bias = params[9]
    
    # Compute activations
    # Add small epsilon to prevent division by zero if we were doing softmax, but tanh is safe.
    a1 = np.tanh(w1 * x + b1)
    a2 = np.tanh(w2 * x + b2)
    a3 = np.tanh(w3 * x + b3)
    
    # Linear combination
    prediction = bias + c1 * a1 + c2 * a2 + c3 * a3
    
    # Ensure finite values
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
