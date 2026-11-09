"""Symbolic regression with 1 input(s) using Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = params
    
    # Define 10 centers uniformly spaced in [0.1, 6.0]
    # This covers the input domain well.
    # Using linspace ensures coverage of the entire range.
    centers = np.linspace(0.1, 6.0, 10)
    
    # Gaussian width. 
    # A width of 0.5 allows local adaptation. 
    # If the function is very smooth, a wider width might be better, 
    # but local flexibility is generally safer for unknown functions 
    # than global basis functions which can suffer from Runge's phenomenon 
    # or lack of local detail.
    sigma = 0.5
    
    # Compute RBF values: exp(-(x - c)^2 / (2 * sigma^2))
    # We can vectorize this: 
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination: rbf_values (N, 10) dot p (10,) -> (N,)
    pred = rbf_values.dot(p)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
