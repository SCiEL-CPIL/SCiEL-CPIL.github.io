"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a 1-hidden-layer neural network with 3 tanh neurons.
    
    Parameters (length 10):
    [a1, b1, w1, a2, b2, w2, a3, b3, w3, c]
    
    Model:
    y = w1 * tanh(a1 * x + b1) + w2 * tanh(a2 * x + b2) + w3 * tanh(a3 * x + b3) + c
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    params = np.asarray(params, dtype=float)
    if params.shape[0] != 10:
        # Fallback to safe ones if shape mismatch (should not happen)
        params = np.ones(10)
        
    # Unpack parameters
    a1, b1, w1 = params[0], params[1], params[2]
    a2, b2, w2 = params[3], params[4], params[5]
    a3, b3, w3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute hidden layer activations
    # Clip pre-activations to avoid numerical issues in gradient calculation 
    # though tanh is safe, it helps with extreme values during optimization
    z1 = np.clip(a1 * x + b1, -50, 50)
    z2 = np.clip(a2 * x + b2, -50, 50)
    z3 = np.clip(a3 * x + b3, -50, 50)
    
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    h3 = np.tanh(z3)
    
    # Output
    prediction = w1 * h1 + w2 * h2 + w3 * h3 + c
    
    # Ensure finite values
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
