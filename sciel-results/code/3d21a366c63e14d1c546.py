import numpy as np
import time

# Module level time tracking for safety
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF network with 9 fixed centers and 1 learnable width parameter
    # Centers span the input domain [0.1, 6]
    centers = np.array([0.1 + i * (6.0 - 0.1) / 8.0 for i in range(9)])
    
    # params[0] is the learnable width (sigma)
    # params[1:10] are the weights for each RBF
    sigma = params[0]
    weights = params[1:10]
    
    # Ensure sigma is positive and reasonable
    sigma = np.clip(sigma, 0.05, 5.0)
    
    # Compute RBF values: exp(-((x - center)^2) / (2 * sigma^2))
    # x has shape (N,), centers has shape (9,)
    # We need to compute (x_i - center_j)^2 for all i, j
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # shape (N, 9)
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))  # shape (N, 9)
    
    # Weighted sum of RBFs
    y = rbf_values @ weights
    
    return y
