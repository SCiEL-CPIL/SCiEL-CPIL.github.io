"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    1-2-1 Neural Network.
    Input: x, y (each shape (N,))
    Hidden layer: 2 neurons with tanh activation.
    Output: Linear combination of hidden units + bias.
    
    Parameters (10 total):
    p0: weight x -> h1
    p1: weight y -> h1
    p2: bias h1
    p3: weight x -> h2
    p4: weight y -> h2
    p5: bias h2
    p6: weight h1 -> out
    p7: weight h2 -> out
    p8: bias out
    p9: unused (kept for compatibility, set to 0 in effect)
    """
    if params is None or len(params) < 9:
        return np.zeros_like(x)
    
    # Extract weights
    wx1, wy1, b1 = params[0], params[1], params[2]
    wx2, wy2, b2 = params[3], params[4], params[5]
    wh1, wh2, bo = params[6], params[7], params[8]
    
    # Compute hidden layer activations
    h1 = np.tanh(wx1 * x + wy1 * y + b1)
    h2 = np.tanh(wx2 * x + wy2 * y + b2)
    
    # Compute output
    return wh1 * h1 + wh2 * h2 + bo
# EVOLVE-BLOCK-END
