"""Rewritten program for symbolic regression with a Mixture of Logistic (Sigmoid) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a basis of linear terms and logistic sigmoid gates.
    Basis: [1, x, y, z, sigma(x), sigma(y), sigma(z), sigma(x+y), sigma(x+z), sigma(y+z)]
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Helper for stable sigmoid
    def sigmoid(t):
        # Clip to avoid overflow in exp
        t = np.clip(t, -500, 500)
        return 1.0 / (1.0 + np.exp(-t))
    
    # Compute basis terms
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = y
    term_3 = z
    
    # Sigmoidal terms with scaling to make them active in the domain [0.5, 2]
    # Center is roughly 1.25. Scaling factor 4 makes it steep enough.
    scale = 4.0
    center = 1.25
    
    term_4 = sigmoid(scale * (x - center))
    term_5 = sigmoid(scale * (y - center))
    term_6 = sigmoid(scale * (z - center))
    
    # Interaction sigmoids
    term_7 = sigmoid(scale * (x + y - 2 * center))
    term_8 = sigmoid(scale * (x + z - 2 * center))
    term_9 = sigmoid(scale * (y + z - 2 * center))
    
    # Combine with parameters
    # params[0] * term_0 + ... + params[9] * term_9
    prediction = (
        params[0] * term_0 +
        params[1] * term_1 +
        params[2] * term_2 +
        params[3] * term_3 +
        params[4] * term_4 +
        params[5] * term_5 +
        params[6] * term_6 +
        params[7] * term_7 +
        params[8] * term_8 +
        params[9] * term_9
    )
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(prediction)):
        # Return large values to penalize during fitting if it happens
        return np.full_like(prediction, 1e30)
        
    return prediction
# EVOLVE-BLOCK-END
